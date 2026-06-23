// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use anyhow::anyhow;
use datafusion::arrow::array::Int32Array;
use datafusion::arrow::array::RecordBatch;
use datafusion::arrow::datatypes::DataType;
use datafusion::arrow::util::pretty::pretty_format_batches;
use datafusion::datasource::provider::DefaultTableFactory;
use datafusion::execution::SessionStateBuilder;
use datafusion::prelude::SessionConfig;
use datafusion::prelude::SessionContext;
use datafusion_common::GetExt;
use datafusion_physical_plan::display::DisplayableExecutionPlan;
use insta::assert_snapshot;
use object_store::ObjectStore;
use object_store::memory::InMemory;
use rstest::rstest;
use vortex::VortexSessionDefault;
use vortex::array::IntoArray;
use vortex::array::arrays::ChunkedArray;
use vortex::array::arrays::StructArray;
use vortex::array::arrays::VarBinArray;
use vortex::array::validity::Validity;
use vortex::buffer::Buffer;
use vortex::buffer::buffer;
use vortex::file::OpenOptionsSessionExt;
use vortex::file::WriteOptionsSessionExt;
use vortex::io::VortexWrite;
use vortex::io::object_store::ObjectStoreReadAt;
use vortex::io::object_store::ObjectStoreWrite;
use vortex::io::runtime::Handle;
use vortex::layout::LayoutStrategy;
use vortex::layout::layouts::chunked::writer::ChunkedLayoutStrategy;
use vortex::layout::layouts::flat::writer::FlatLayoutStrategy;
use vortex::layout::layouts::table::TableStrategy;
use vortex::session::VortexSession;

use crate::VortexFormatFactory;
use crate::common_tests::TestSessionContext;

fn make_session(
    object_store: Arc<dyn ObjectStore>,
    repartition_file_scans: bool,
) -> SessionContext {
    let factory = Arc::new(VortexFormatFactory::new());

    let config = SessionConfig::new()
        .with_target_partitions(4)
        .with_repartition_file_scans(repartition_file_scans)
        .with_repartition_file_min_size(0);
    let mut state = SessionStateBuilder::new()
        .with_config(config)
        .with_default_features()
        .with_table_factory(
            factory.get_ext().to_uppercase(),
            Arc::new(DefaultTableFactory::new()),
        )
        .with_object_store(&url::Url::try_from("file://").unwrap(), object_store);

    if let Some(file_formats) = state.file_formats() {
        file_formats.push(factory as _);
    }

    SessionContext::new_with_state(state.build()).enable_url_table()
}

async fn count_query_partitions(ctx: &SessionContext, sql: &str) -> anyhow::Result<usize> {
    let explain = ctx.sql(&format!("EXPLAIN {sql}")).await?.collect().await?;
    let plan = pretty_format_batches(&explain)?.to_string();
    let marker = "DataSourceExec: file_groups={";
    let start = plan
        .find(marker)
        .ok_or_else(|| anyhow!("EXPLAIN plan did not contain a DataSourceExec"))?
        + marker.len();
    let partitions = plan[start..]
        .chars()
        .take_while(|ch| ch.is_ascii_digit())
        .collect::<String>();

    Ok(partitions.parse()?)
}

fn batch_values(batches: &[RecordBatch]) -> Vec<i32> {
    let mut values = Vec::with_capacity(batches.iter().map(|batch| batch.num_rows()).sum());

    for batch in batches {
        let array = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .expect("value column should be Int32");
        values.extend(array.values().iter().copied());
    }

    values
}

#[rstest]
#[tokio::test]
async fn test_query_file(#[values(Some(1), None)] limit: Option<usize>) -> anyhow::Result<()> {
    let ctx = TestSessionContext::default();

    let session = VortexSession::default();

    let strings = ChunkedArray::from_iter([
        VarBinArray::from(vec!["ab", "foo", "bar", "baz"]).into_array(),
        VarBinArray::from(vec!["ab", "foo", "bar", "baz"]).into_array(),
    ])
    .into_array();

    let numbers = ChunkedArray::from_iter([
        buffer![1u32, 2, 3, 4].into_array(),
        buffer![5u32, 6, 7, 8].into_array(),
    ])
    .into_array();

    let st = StructArray::try_new(
        ["strings", "numbers"].into(),
        vec![strings, numbers],
        8,
        Validity::NonNullable,
    )?;

    let mut writer = ObjectStoreWrite::new(Arc::clone(&ctx.store), &"test.vortex".into()).await?;

    let summary = session
        .write_options()
        .write(&mut writer, st.into_array().to_array_stream())
        .await?;

    writer.shutdown().await?;

    assert_eq!(summary.row_count(), 8);

    let read_row_count = ctx
        .session
        .sql("SELECT * from '/test.vortex'")
        .await?
        .limit(0, limit)?
        .count()
        .await?;

    assert_eq!(read_row_count, limit.unwrap_or(8));

    Ok(())
}

#[tokio::test]
async fn test_addition_pushdown() -> anyhow::Result<()> {
    let ctx = TestSessionContext::default();

    ctx.session
        .sql(
            "CREATE EXTERNAL TABLE written_data \
                    (a TINYINT NOT NULL) \
                STORED AS vortex \
                LOCATION '/test/'",
        )
        .await?;

    ctx.session
        .sql("INSERT INTO written_data VALUES (0), (1), (2), (3), (4)")
        .await?
        .collect()
        .await?;

    let result = ctx
        .session
        .sql("SELECT a, a + 5 as five, a + 6 as six FROM written_data WHERE a + 5 > 7")
        .await?
        .collect()
        .await?;

    assert_snapshot!(pretty_format_batches(&result)?, @r"
        +---+------+-----+
        | a | five | six |
        +---+------+-----+
        | 3 | 8    | 9   |
        | 4 | 9    | 10  |
        +---+------+-----+
        ");

    Ok(())
}

#[tokio::test]
async fn test_octet_length_pushdown() -> anyhow::Result<()> {
    let ctx = TestSessionContext::new(true);

    ctx.session
        .sql(
            "CREATE EXTERNAL TABLE written_strings \
                    (s VARCHAR NOT NULL) \
                STORED AS vortex \
                LOCATION '/strings/'",
        )
        .await?;

    ctx.session
        .sql("INSERT INTO written_strings VALUES ('a'), ('é'), ('abcd'), ('')")
        .await?
        .collect()
        .await?;

    let result = ctx
        .session
        .sql(
            "SELECT s, octet_length(s) AS len \
             FROM written_strings \
             WHERE octet_length(s) > 1 \
             ORDER BY s",
        )
        .await?
        .collect()
        .await?;

    assert_eq!(
        result[0].schema().field_with_name("len")?.data_type(),
        &DataType::Int32
    );
    assert_snapshot!(pretty_format_batches(&result)?, @r"
        +------+-----+
        | s    | len |
        +------+-----+
        | abcd | 4   |
        | é    | 2   |
        +------+-----+
        ");

    Ok(())
}

#[tokio::test]
async fn create_table_ordered_by() -> anyhow::Result<()> {
    let ctx = TestSessionContext::default();

    // Vortex
    ctx.session
        .sql(
            "CREATE EXTERNAL TABLE my_tbl_vx \
                (c1 VARCHAR NOT NULL, c2 INT NOT NULL) \
                STORED AS vortex  \
                WITH ORDER (c1 ASC)
                LOCATION '/test/'",
        )
        .await?;

    ctx.session
        .sql("INSERT INTO my_tbl_vx VALUES ('air', 5), ('balloon', 42)")
        .await?
        .collect()
        .await?;

    ctx.session
        .sql("INSERT INTO my_tbl_vx VALUES ('zebra', 5)")
        .await?
        .collect()
        .await?;

    ctx.session
        .sql("INSERT INTO my_tbl_vx VALUES ('texas', 2000), ('alabama', 2000)")
        .await?
        .collect()
        .await?;

    let df = ctx
        .session
        .sql("SELECT * FROM my_tbl_vx ORDER BY c1 ASC limit 3")
        .await?;

    let physical_plan = ctx
        .session
        .state()
        .create_physical_plan(df.logical_plan())
        .await?;

    insta::assert_snapshot!(DisplayableExecutionPlan::new(physical_plan.as_ref())
                .tree_render().to_string(), @r"
        ┌───────────────────────────┐
        │  SortPreservingMergeExec  │
        │    --------------------   │
        │     c1 ASC NULLS LAST     │
        │                           │
        │          limit: 3         │
        └─────────────┬─────────────┘
        ┌─────────────┴─────────────┐
        │       DataSourceExec      │
        │    --------------------   │
        │          files: 3         │
        │       format: vortex      │
        └───────────────────────────┘
        ");

    let r = df.collect().await?;

    insta::assert_snapshot!(pretty_format_batches(&r)?.to_string(), @r"
        +---------+------+
        | c1      | c2   |
        +---------+------+
        | air     | 5    |
        | alabama | 2000 |
        | balloon | 42   |
        +---------+------+
        ");

    Ok(())
}

/// Doc example: demonstrates creating, writing, reading, and filtering a Vortex table.
#[tokio::test]
async fn doc_example() -> anyhow::Result<()> {
    // [setup]
    use std::sync::Arc;

    use datafusion::datasource::provider::DefaultTableFactory;
    use datafusion::execution::SessionStateBuilder;
    use datafusion::prelude::SessionContext;
    use datafusion_common::GetExt;
    use object_store::memory::InMemory;

    use crate::VortexFormatFactory;

    let factory = Arc::new(VortexFormatFactory::new());
    let state = SessionStateBuilder::new()
        .with_default_features()
        .with_table_factory(
            factory.get_ext().to_uppercase(),
            Arc::new(DefaultTableFactory::new()),
        )
        .with_file_formats(vec![factory])
        .build();
    let ctx = SessionContext::new_with_state(state).enable_url_table();
    // [setup]

    // Register an in-memory object store for the test.
    let store = Arc::new(InMemory::new());
    ctx.register_object_store(&url::Url::try_from("file://").unwrap(), store);

    // [create]
    ctx.sql(
        "CREATE EXTERNAL TABLE my_table \
                (name VARCHAR NOT NULL, age INT NOT NULL) \
            STORED AS vortex \
            LOCATION '/demo/'",
    )
    .await?;
    // [create]

    // [write]
    ctx.sql(
        "INSERT INTO my_table VALUES \
                ('Alice', 30), ('Bob', 25), ('Charlie', 35), ('Diana', 28)",
    )
    .await?
    .collect()
    .await?;
    // [write]

    // [query]
    let result = ctx
        .sql("SELECT name, age FROM my_table WHERE age > 28 ORDER BY age")
        .await?
        .collect()
        .await?;
    // [query]

    assert_snapshot!(pretty_format_batches(&result)?, @r"
        +---------+-----+
        | name    | age |
        +---------+-----+
        | Alice   | 30  |
        | Charlie | 35  |
        +---------+-----+
        ");

    Ok(())
}

#[tokio::test]
async fn test_repartitioned_scan_matches_non_repartitioned_for_uneven_splits() -> anyhow::Result<()>
{
    let store = Arc::new(InMemory::new()) as _;
    let session = VortexSession::default();
    let path = object_store::path::Path::parse("/split-aligned-repartition.vortex")?;

    let chunk_1_len = 2_000;
    let chunk_2_len = 5_000;
    let chunk_3_len = 6_000;
    let row_count = chunk_1_len + chunk_2_len + chunk_3_len;

    let chunk_1 = StructArray::try_new(
        ["value"].into(),
        vec![Buffer::from_iter(0_i32..chunk_1_len).into_array()],
        usize::try_from(chunk_1_len)?,
        Validity::NonNullable,
    )?;
    let chunk_2 = StructArray::try_new(
        ["value"].into(),
        vec![Buffer::from_iter(chunk_1_len..(chunk_1_len + chunk_2_len)).into_array()],
        usize::try_from(chunk_2_len)?,
        Validity::NonNullable,
    )?;
    let chunk_3 = StructArray::try_new(
        ["value"].into(),
        vec![Buffer::from_iter((chunk_1_len + chunk_2_len)..row_count).into_array()],
        usize::try_from(chunk_3_len)?,
        Validity::NonNullable,
    )?;
    let table = ChunkedArray::from_iter([
        chunk_1.into_array(),
        chunk_2.into_array(),
        chunk_3.into_array(),
    ])
    .into_array();
    let flat: Arc<dyn LayoutStrategy> = Arc::new(FlatLayoutStrategy::default());
    let strategy: Arc<dyn LayoutStrategy> = Arc::new(TableStrategy::new(
        Arc::clone(&flat),
        Arc::new(ChunkedLayoutStrategy::new(FlatLayoutStrategy::default())),
    ));

    let mut writer = ObjectStoreWrite::new(Arc::clone(&store), &path).await?;
    let summary = session
        .write_options()
        .with_strategy(strategy)
        .write(&mut writer, table.into_array().to_array_stream())
        .await?;
    writer.shutdown().await?;

    let reader = Arc::new(ObjectStoreReadAt::new(
        Arc::clone(&store),
        path.clone(),
        Handle::find().expect("tokio runtime should be available in tests"),
    ));
    let vxf = session
        .open_options()
        .with_file_size(summary.size())
        .open_read(reader)
        .await?;
    let split_ranges = vxf.splits()?;
    let split_lengths = split_ranges
        .iter()
        .map(|range| range.end - range.start)
        .collect::<Vec<_>>();

    assert!(split_ranges.len() > 1);
    assert!(
        split_lengths
            .windows(2)
            .any(|window| window[0] != window[1])
    );

    let serial_ctx = make_session(Arc::clone(&store), false);
    let repartitioned_ctx = make_session(Arc::clone(&store), true);
    let repartitioned_partitions = count_query_partitions(
        &repartitioned_ctx,
        "SELECT value FROM '/split-aligned-repartition.vortex'",
    )
    .await?;

    assert!(repartitioned_partitions > 1);

    let serial = serial_ctx
        .sql("SELECT value FROM '/split-aligned-repartition.vortex' ORDER BY value")
        .await?
        .collect()
        .await?;
    let repartitioned = repartitioned_ctx
        .sql("SELECT value FROM '/split-aligned-repartition.vortex' ORDER BY value")
        .await?
        .collect()
        .await?;
    let serial_values = batch_values(&serial);
    let repartitioned_values = batch_values(&repartitioned);
    let expected = (0_i32..row_count).collect::<Vec<_>>();

    assert_eq!(serial_values, expected);
    assert_eq!(repartitioned_values, serial_values);

    Ok(())
}

/// Roundtrip an `arrow.uuid` extension column through a Vortex file: write the column directly
/// via the session-aware Arrow→Vortex conversion, then `SELECT *` and assert both the field
/// metadata and the underlying values survive the trip.
#[tokio::test]
async fn arrow_uuid_extension_roundtrip() -> anyhow::Result<()> {
    use arrow_schema::DataType;
    use arrow_schema::Field;
    use arrow_schema::Schema;
    use arrow_schema::extension::Uuid;
    use datafusion::arrow::array::FixedSizeBinaryArray;
    use datafusion::arrow::array::RecordBatch;
    use datafusion::assert_batches_sorted_eq;
    use vortex_arrow::ArrowSessionExt;

    let ctx = TestSessionContext::default();
    // Default vortex session has importer/exporter for Arrow UUID
    let session = VortexSession::default();

    let mut uuid_field = Field::new("id", DataType::FixedSizeBinary(16), false);
    uuid_field.try_with_extension_type(Uuid)?;
    let schema = Arc::new(Schema::new(vec![uuid_field]));

    let uuids = FixedSizeBinaryArray::try_from_iter(
        [*b"0123456789abcdef", *b"fedcba9876543210"].into_iter(),
    )?;
    let batch = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(uuids)])?;
    let array = session.arrow().from_arrow_record_batch(batch, &schema)?;

    let mut writer = ObjectStoreWrite::new(Arc::clone(&ctx.store), &"uuid.vortex".into()).await?;
    session
        .write_options()
        .write(&mut writer, array.to_array_stream())
        .await?;
    writer.shutdown().await?;

    let result = ctx
        .session
        .sql("SELECT * FROM '/uuid.vortex'")
        .await?
        .collect()
        .await?;

    assert!(
        result[0]
            .schema_ref()
            .field(0)
            .has_valid_extension_type::<Uuid>()
    );

    assert_batches_sorted_eq!(
        [
            "+----------------------------------+",
            "| id                               |",
            "+----------------------------------+",
            "| 30313233343536373839616263646566 |",
            "| 66656463626139383736353433323130 |",
            "+----------------------------------+",
        ],
        &result
    );

    Ok(())
}

/// Same as [`arrow_uuid_extension_roundtrip`] but with the `arrow.uuid` field nested inside a
/// top-level `Struct`, exercising recursive session-aware Field/Schema inference: if any layer
/// falls back to the non-plugin canonical path, the inner field loses its extension metadata.
#[tokio::test]
async fn arrow_uuid_extension_roundtrip_nested_struct() -> anyhow::Result<()> {
    use arrow_schema::DataType;
    use arrow_schema::Field;
    use arrow_schema::Fields;
    use arrow_schema::Schema;
    use arrow_schema::extension::Uuid;
    use datafusion::arrow::array::Array;
    use datafusion::arrow::array::FixedSizeBinaryArray;
    use datafusion::arrow::array::RecordBatch;
    use datafusion::arrow::array::StructArray as ArrowStructArray;
    use datafusion::assert_batches_sorted_eq;
    use vortex_arrow::ArrowSessionExt;

    let ctx = TestSessionContext::default();
    let session = VortexSession::default();

    let mut inner_uuid_field = Field::new("id", DataType::FixedSizeBinary(16), false);
    inner_uuid_field.try_with_extension_type(Uuid)?;
    let payload_fields = Fields::from(vec![inner_uuid_field]);
    let payload_field = Field::new("payload", DataType::Struct(payload_fields.clone()), false);
    let schema = Arc::new(Schema::new(vec![payload_field]));

    let uuids: Arc<dyn Array> = Arc::new(FixedSizeBinaryArray::try_from_iter(
        [*b"0123456789abcdef", *b"fedcba9876543210"].into_iter(),
    )?);
    let payload_array = ArrowStructArray::new(payload_fields, vec![uuids], None);
    let batch = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(payload_array)])?;
    let array = session.arrow().from_arrow_record_batch(batch, &schema)?;

    let mut writer =
        ObjectStoreWrite::new(Arc::clone(&ctx.store), &"uuid_struct.vortex".into()).await?;
    session
        .write_options()
        .write(&mut writer, array.to_array_stream())
        .await?;
    writer.shutdown().await?;

    let result = ctx
        .session
        .sql("SELECT payload FROM '/uuid_struct.vortex'")
        .await?
        .collect()
        .await?;

    let read_payload = result[0].schema_ref().field(0);
    let DataType::Struct(read_inner) = read_payload.data_type() else {
        panic!(
            "expected Struct payload, got {:?}",
            read_payload.data_type()
        );
    };
    assert!(read_inner[0].has_valid_extension_type::<Uuid>());

    assert_batches_sorted_eq!(
        [
            "+----------------------------------------+",
            "| payload                                |",
            "+----------------------------------------+",
            "| {id: 30313233343536373839616263646566} |",
            "| {id: 66656463626139383736353433323130} |",
            "+----------------------------------------+",
        ],
        &result
    );

    Ok(())
}

/// Build a single-partition session with dynamic-filter pushdown toggleable, for join tests.
fn join_ctx(store: Arc<dyn ObjectStore>, dynamic_filter_pushdown: bool) -> SessionContext {
    let factory = Arc::new(VortexFormatFactory::new());
    let mut config = SessionConfig::new().with_target_partitions(1);
    config
        .options_mut()
        .optimizer
        .enable_dynamic_filter_pushdown = dynamic_filter_pushdown;
    let mut state = SessionStateBuilder::new()
        .with_config(config)
        .with_default_features()
        .with_table_factory(
            factory.get_ext().to_uppercase(),
            Arc::new(DefaultTableFactory::new()),
        )
        .with_object_store(&url::Url::try_from("file://").unwrap(), store);
    if let Some(file_formats) = state.file_formats() {
        file_formats.push(factory as _);
    }
    SessionContext::new_with_state(state.build()).enable_url_table()
}

/// Extract the `output_rows=` value (e.g. "2" or "10.00 K") for the `DataSourceExec` line scanning
/// the file named `file_substr`, from a pretty-printed `EXPLAIN ANALYZE` plan.
fn probe_scan_output_rows(plan: &str, file_substr: &str) -> anyhow::Result<String> {
    let line = plan
        .lines()
        .find(|l| l.contains("DataSourceExec") && l.contains(file_substr))
        .ok_or_else(|| anyhow!("no DataSourceExec for {file_substr} in:\n{plan}"))?;
    let start = line
        .find("output_rows=")
        .ok_or_else(|| anyhow!("no output_rows in: {line}"))?
        + "output_rows=".len();
    Ok(line[start..]
        .split(',')
        .next()
        .unwrap_or("")
        .trim()
        .to_string())
}

/// ACCEPTANCE (red until Step 4): a small build side makes DataFusion emit a selective dynamic
/// `InList` (k IN {2,7000}). Once we snapshot + route it into the Vortex scan, the probe-side scan
/// must emit only the 2 matching rows (today it scans all 10k). Also guards #4144: results identical
/// with dynamic-filter pushdown on and off (we report Unsupported to DF, so the join still filters).
#[tokio::test]
async fn dynamic_inlist_filter_applied_to_probe_scan() -> anyhow::Result<()> {
    let store = Arc::new(InMemory::new()) as Arc<dyn ObjectStore>;
    let session = VortexSession::default();

    // fact / probe side: 10k rows across 2 chunks (multiple zones).
    let fact = StructArray::try_new(
        ["k", "v"].into(),
        vec![
            ChunkedArray::from_iter([
                Buffer::from_iter(0_i32..5_000).into_array(),
                Buffer::from_iter(5_000_i32..10_000).into_array(),
            ])
            .into_array(),
            ChunkedArray::from_iter([
                Buffer::from_iter(0_i32..5_000).into_array(),
                Buffer::from_iter(5_000_i32..10_000).into_array(),
            ])
            .into_array(),
        ],
        10_000,
        Validity::NonNullable,
    )?;
    {
        let mut w = ObjectStoreWrite::new(Arc::clone(&store), &"fact.vortex".into()).await?;
        session
            .write_options()
            .write(&mut w, fact.into_array().to_array_stream())
            .await?;
        w.shutdown().await?;
    }

    // dim / build side: 2 selective keys (one per fact chunk); 2 distinct < InList threshold (150).
    let dim = StructArray::try_new(
        ["k", "label"].into(),
        vec![
            buffer![2_i32, 7_000].into_array(),
            buffer![100_i32, 100].into_array(),
        ],
        2,
        Validity::NonNullable,
    )?;
    {
        let mut w = ObjectStoreWrite::new(Arc::clone(&store), &"dim.vortex".into()).await?;
        session
            .write_options()
            .write(&mut w, dim.into_array().to_array_stream())
            .await?;
        w.shutdown().await?;
    }

    let query = "SELECT f.k FROM '/fact.vortex' f JOIN '/dim.vortex' d ON f.k = d.k ORDER BY f.k";

    // Correctness + #4144 guard: identical results with dynamic-filter pushdown on and off.
    let on = join_ctx(Arc::clone(&store), true)
        .sql(query)
        .await?
        .collect()
        .await?;
    let off = join_ctx(Arc::clone(&store), false)
        .sql(query)
        .await?
        .collect()
        .await?;
    assert_eq!(
        batch_values(&on),
        vec![2, 7000],
        "join result with pushdown on"
    );
    assert_eq!(
        batch_values(&off),
        batch_values(&on),
        "dynamic-filter pushdown must not change results"
    );

    // Effect: the probe scan emits only the matching rows because the dynamic InList is routed
    // into the Vortex scan (it scanned all 10k before this feature).
    let explain = join_ctx(Arc::clone(&store), true)
        .sql(&format!("EXPLAIN ANALYZE {query}"))
        .await?
        .collect()
        .await?;
    let plan = pretty_format_batches(&explain)?.to_string();
    assert_eq!(
        probe_scan_output_rows(&plan, "fact.vortex")?,
        "2",
        "probe scan should emit only the 2 matching rows; plan:\n{plan}"
    );

    Ok(())
}

/// Confirm the dynamic InList still produces correct results under a multi-partition, file-scan
/// repartitioned plan (`make_session(.., true)` → `target_partitions(4)` + repartition) — closer to
/// real execution than the single-partition determinism tests above, and exercising the
/// repartition path that #18513 concerned.
#[tokio::test]
async fn dynamic_inlist_correct_multi_partition() -> anyhow::Result<()> {
    let store = Arc::new(InMemory::new()) as Arc<dyn ObjectStore>;
    let session = VortexSession::default();

    let fact = StructArray::try_new(
        ["k", "v"].into(),
        vec![
            ChunkedArray::from_iter([
                Buffer::from_iter(0_i32..5_000).into_array(),
                Buffer::from_iter(5_000_i32..10_000).into_array(),
            ])
            .into_array(),
            ChunkedArray::from_iter([
                Buffer::from_iter(0_i32..5_000).into_array(),
                Buffer::from_iter(5_000_i32..10_000).into_array(),
            ])
            .into_array(),
        ],
        10_000,
        Validity::NonNullable,
    )?;
    {
        let mut w = ObjectStoreWrite::new(Arc::clone(&store), &"fact.vortex".into()).await?;
        session
            .write_options()
            .write(&mut w, fact.into_array().to_array_stream())
            .await?;
        w.shutdown().await?;
    }

    let dim = StructArray::try_new(
        ["k", "label"].into(),
        vec![
            buffer![2_i32, 7_000].into_array(),
            buffer![100_i32, 100].into_array(),
        ],
        2,
        Validity::NonNullable,
    )?;
    {
        let mut w = ObjectStoreWrite::new(Arc::clone(&store), &"dim.vortex".into()).await?;
        session
            .write_options()
            .write(&mut w, dim.into_array().to_array_stream())
            .await?;
        w.shutdown().await?;
    }

    // 4 partitions + file-scan repartitioning.
    let ctx = make_session(Arc::clone(&store), true);
    let result = ctx
        .sql("SELECT f.k FROM '/fact.vortex' f JOIN '/dim.vortex' d ON f.k = d.k ORDER BY f.k")
        .await?
        .collect()
        .await?;
    assert_eq!(
        batch_values(&result),
        vec![2, 7000],
        "multi-partition join result must match"
    );

    Ok(())
}

/// Scope guard: a TopK query (`ORDER BY .. LIMIT k`) pushes a dynamic `col < threshold` bound, not
/// an `InList`. Our InList-only in-scan routing must leave it untouched (it stays a file-level
/// prune via `FilePruner`), so TopK must still return correct results with the feature active.
#[tokio::test]
async fn topk_dynamic_threshold_not_pushed_in_scan() -> anyhow::Result<()> {
    let store = Arc::new(InMemory::new()) as Arc<dyn ObjectStore>;
    let session = VortexSession::default();

    let fact = StructArray::try_new(
        ["k"].into(),
        vec![
            ChunkedArray::from_iter([
                Buffer::from_iter(0_i32..5_000).into_array(),
                Buffer::from_iter(5_000_i32..10_000).into_array(),
            ])
            .into_array(),
        ],
        10_000,
        Validity::NonNullable,
    )?;
    {
        let mut w = ObjectStoreWrite::new(Arc::clone(&store), &"fact.vortex".into()).await?;
        session
            .write_options()
            .write(&mut w, fact.into_array().to_array_stream())
            .await?;
        w.shutdown().await?;
    }

    // Dynamic filter pushdown on → the TopK threshold reaches the scan's pruning predicate.
    let ctx = join_ctx(Arc::clone(&store), true);
    let result = ctx
        .sql("SELECT k FROM '/fact.vortex' ORDER BY k ASC LIMIT 5")
        .await?
        .collect()
        .await?;
    assert_eq!(
        batch_values(&result),
        vec![0, 1, 2, 3, 4],
        "TopK must return the 5 smallest keys with the feature active"
    );

    Ok(())
}

/// Dynamic-path adapter-rewrite under schema evolution: the probe file stores the join key `k` as
/// `Int16`, but the table declares it `INT` (`Int32`). The dynamic `InList` (built from `Int32` dim
/// keys) must be snapshotted and adapter-rewritten to the file's physical type, then still applied
/// in-scan — not silently dropped or errored. Asserts correct results AND that the probe scan emits
/// only the matching rows (so the rewrite genuinely fired rather than being dropped).
#[tokio::test]
async fn dynamic_inlist_applied_under_schema_evolution() -> anyhow::Result<()> {
    let store = Arc::new(InMemory::new()) as Arc<dyn ObjectStore>;
    let session = VortexSession::default();

    // fact / probe: join key `k` physically Int16 (the table declares it INT → scan upcasts).
    let fact = StructArray::try_new(
        ["k", "v"].into(),
        vec![
            buffer![0i16, 3, 5, 7, 8, 9].into_array(),
            buffer![0i32, 30, 50, 70, 80, 90].into_array(),
        ],
        6,
        Validity::NonNullable,
    )?;
    {
        let mut w = ObjectStoreWrite::new(Arc::clone(&store), &"factdir/f1.vortex".into()).await?;
        session
            .write_options()
            .write(&mut w, fact.into_array().to_array_stream())
            .await?;
        w.shutdown().await?;
    }

    // dim / build: 2 selective Int32 keys → dynamic InList {3, 7}.
    let dim = StructArray::try_new(
        ["k"].into(),
        vec![buffer![3i32, 7].into_array()],
        2,
        Validity::NonNullable,
    )?;
    {
        let mut w = ObjectStoreWrite::new(Arc::clone(&store), &"dimdir/d1.vortex".into()).await?;
        session
            .write_options()
            .write(&mut w, dim.into_array().to_array_stream())
            .await?;
        w.shutdown().await?;
    }

    let ctx = join_ctx(Arc::clone(&store), true);
    ctx.sql("CREATE EXTERNAL TABLE fact (k INT, v INT) STORED AS vortex LOCATION '/factdir/'")
        .await?;
    ctx.sql("CREATE EXTERNAL TABLE dim (k INT) STORED AS vortex LOCATION '/dimdir/'")
        .await?;

    let query = "SELECT f.k FROM fact f JOIN dim d ON f.k = d.k ORDER BY f.k";
    let result = ctx.sql(query).await?.collect().await?;
    assert_eq!(
        batch_values(&result),
        vec![3, 7],
        "join result under Int16->Int32 schema evolution"
    );

    let explain = ctx
        .sql(&format!("EXPLAIN ANALYZE {query}"))
        .await?
        .collect()
        .await?;
    let plan = pretty_format_batches(&explain)?.to_string();
    assert_eq!(
        probe_scan_output_rows(&plan, "f1.vortex")?,
        "2",
        "dynamic InList must apply in-scan despite schema evolution; plan:\n{plan}"
    );

    Ok(())
}

/// MEASUREMENT (compare `VORTEX_DYNAMIC_INSCAN` on vs off): the in-scan InList's I/O win where it
/// fires. The key `k` is sorted (clustered) so the zoned layout's coarse zones get tight min/max;
/// `v` is a high-entropy payload (expensive to read) and is what the query projects. The dim's keys
/// fall only in the first coarse zone, so the in-scan InList prunes the other zones and skips their
/// payload reads (file-level pruning can't — the one file's stats span the whole key range).
#[tokio::test]
async fn measure_dynamic_inlist_bytes_read() -> anyhow::Result<()> {
    use datafusion::physical_plan::collect;

    use crate::persistent::metrics::VortexMetricsFinder;

    let store = Arc::new(InMemory::new()) as Arc<dyn ObjectStore>;
    let session = VortexSession::default();

    // 1M rows. `k` is sorted/contiguous → clustered, so the layout's coarse zones get tight min/max
    // and the in-scan InList can prune them. `v` is a high-entropy payload (expensive to read) and
    // is what we project, so pruning skips its reads for pruned zones. The dim keys all fall in the
    // first coarse zone.
    let n_chunks = 100_i32;
    let chunk_len = 10_000_i32;
    let k_chunks: Vec<_> = (0..n_chunks)
        .map(|c| Buffer::from_iter((c * chunk_len)..(c * chunk_len + chunk_len)).into_array())
        .collect();
    let v_chunks: Vec<_> = (0..n_chunks)
        .map(|c| {
            let base = (c * chunk_len) as u32;
            Buffer::from_iter(
                (0..chunk_len as u32)
                    .map(|j| base.wrapping_add(j).wrapping_mul(2_654_435_761) as i32),
            )
            .into_array()
        })
        .collect();
    let fact = StructArray::try_new(
        ["k", "v"].into(),
        vec![
            ChunkedArray::from_iter(k_chunks).into_array(),
            ChunkedArray::from_iter(v_chunks).into_array(),
        ],
        (n_chunks * chunk_len) as usize,
        Validity::NonNullable,
    )?;
    {
        let mut w = ObjectStoreWrite::new(Arc::clone(&store), &"fact.vortex".into()).await?;
        session
            .write_options()
            .write(&mut w, fact.into_array().to_array_stream())
            .await?;
        w.shutdown().await?;
    }

    // 3 keys, all in the first zone (< chunk_len) → InList; the other zones are prunable.
    let dim = StructArray::try_new(
        ["k"].into(),
        vec![buffer![5_i32, 50, 500].into_array()],
        3,
        Validity::NonNullable,
    )?;
    {
        let mut w = ObjectStoreWrite::new(Arc::clone(&store), &"dim.vortex".into()).await?;
        session
            .write_options()
            .write(&mut w, dim.into_array().to_array_stream())
            .await?;
        w.shutdown().await?;
    }

    let ctx = join_ctx(Arc::clone(&store), true);
    let (state, logical) = ctx
        .sql("SELECT f.v FROM '/fact.vortex' f JOIN '/dim.vortex' d ON f.k = d.k")
        .await?
        .into_parts();
    let plan = state.create_physical_plan(&logical).await?;
    let result = collect(Arc::clone(&plan), state.task_ctx()).await?;

    let total_rows: usize = result.iter().map(|b| b.num_rows()).sum();
    let sets = VortexMetricsFinder::find_all(plan.as_ref());
    let enabled = std::env::var("VORTEX_DYNAMIC_INSCAN")
        .map(|v| v != "0")
        .unwrap_or(true);
    println!(
        "MEASURE dynamic_inscan_enabled={enabled} result_rows={total_rows} n_scans={}",
        sets.len()
    );
    for (i, set) in sets.iter().enumerate() {
        for metric in set.aggregate_by_name().sorted_for_display().iter() {
            println!("  scan[{i}] {metric}");
        }
    }

    assert_eq!(total_rows, 3, "join should return the 3 matching rows");
    Ok(())
}

/// GUARD: a large build side (>150 distinct) yields a `hash_lookup` membership Vortex can't
/// represent, leaving only non-selective min/max — which must NOT be pushed per-row. The probe
/// scan must keep reading all rows (no in-scan filter), so this stays green after Step 4 too.
#[tokio::test]
async fn large_build_dynamic_not_pushed_in_scan() -> anyhow::Result<()> {
    let store = Arc::new(InMemory::new()) as Arc<dyn ObjectStore>;
    let session = VortexSession::default();

    // fact / probe side: 10k rows across 2 chunks.
    let fact = StructArray::try_new(
        ["k", "v"].into(),
        vec![
            ChunkedArray::from_iter([
                Buffer::from_iter(0_i32..5_000).into_array(),
                Buffer::from_iter(5_000_i32..10_000).into_array(),
            ])
            .into_array(),
            ChunkedArray::from_iter([
                Buffer::from_iter(0_i32..5_000).into_array(),
                Buffer::from_iter(5_000_i32..10_000).into_array(),
            ])
            .into_array(),
        ],
        10_000,
        Validity::NonNullable,
    )?;
    {
        let mut w = ObjectStoreWrite::new(Arc::clone(&store), &"fact2.vortex".into()).await?;
        session
            .write_options()
            .write(&mut w, fact.into_array().to_array_stream())
            .await?;
        w.shutdown().await?;
    }

    // dim / build side: 300 distinct keys spanning the fact range → >150 distinct, min/max wide.
    let dim = StructArray::try_new(
        ["k", "label"].into(),
        vec![
            Buffer::from_iter((0_i32..300).map(|i| i * 33)).into_array(),
            Buffer::from_iter((0_i32..300).map(|_| 100_i32)).into_array(),
        ],
        300,
        Validity::NonNullable,
    )?;
    {
        let mut w = ObjectStoreWrite::new(Arc::clone(&store), &"dim2.vortex".into()).await?;
        session
            .write_options()
            .write(&mut w, dim.into_array().to_array_stream())
            .await?;
        w.shutdown().await?;
    }

    let explain = join_ctx(Arc::clone(&store), true)
        .sql(
            "EXPLAIN ANALYZE SELECT f.k FROM '/fact2.vortex' f \
             JOIN '/dim2.vortex' d ON f.k = d.k",
        )
        .await?
        .collect()
        .await?;
    let plan = pretty_format_batches(&explain)?.to_string();
    assert_eq!(
        probe_scan_output_rows(&plan, "fact2.vortex")?,
        "10.00 K",
        "large-build probe must not be filtered in-scan (hash_lookup unconvertible); plan:\n{plan}"
    );

    Ok(())
}
