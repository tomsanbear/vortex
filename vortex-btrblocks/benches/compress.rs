// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(clippy::unwrap_used)]

#[cfg(not(codspeed))]
mod benchmarks {
    use std::sync::Arc;
    use std::sync::LazyLock;

    use divan::Bencher;
    use divan::counter::BytesCount;
    use divan::counter::ItemsCount;
    use rand::Rng;
    use rand::SeedableRng;
    use rand::prelude::StdRng;
    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::VarBinViewArray;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::session::ArraySession;
    use vortex_btrblocks::BtrBlocksCompressor;
    use vortex_btrblocks::BtrBlocksCompressorBuilder;
    use vortex_btrblocks::schemes::string::FSSTScheme;
    use vortex_btrblocks::schemes::string::FSSTSchemeWithPretrained;
    use vortex_buffer::buffer_mut;
    use vortex_fsst::fsst_train_compressor;
    use vortex_session::VortexSession;
    use vortex_utils::aliases::hash_set::HashSet;

    static SESSION: LazyLock<VortexSession> =
        LazyLock::new(|| VortexSession::empty().with::<ArraySession>());

    fn make_clickbench_window_name() -> ArrayRef {
        // A test that's meant to mirror the WindowName column from ClickBench.
        let mut values = buffer_mut![-1i32; 65_536];
        let mut visited = HashSet::new();
        let mut rng = StdRng::seed_from_u64(1u64);
        while visited.len() < 223 {
            let random = (rng.next_u32() as usize) % 65_536;
            if visited.contains(&random) {
                continue;
            }
            visited.insert(random);
            // Pick 100 random values to insert.
            values[random] = 5 * (rng.next_u64() % 100) as i32;
        }

        // Ok, now let's compress
        values.freeze().into_array()
    }

    #[divan::bench]
    fn btrblocks(bencher: Bencher) {
        let mut ctx = SESSION.create_execution_ctx();
        let array = make_clickbench_window_name()
            .execute::<PrimitiveArray>(&mut ctx)
            .unwrap();
        let compressor = BtrBlocksCompressor::default();
        bencher
            .with_inputs(|| (&array, SESSION.create_execution_ctx()))
            .input_counter(|(array, _)| ItemsCount::new(array.len()))
            .input_counter(|(array, _)| BytesCount::of_many::<i32>(array.len()))
            .bench_refs(|(array, ctx)| {
                compressor
                    .compress(&array.clone().into_array(), ctx)
                    .unwrap()
            });
    }

    const FSST_FRAGMENT_COUNT: usize = 100;
    const FSST_FRAGMENT_SIZE: usize = 256;

    fn make_string_fragments() -> Vec<ArrayRef> {
        (0..FSST_FRAGMENT_COUNT)
            .map(|batch| {
                let strings: Vec<String> = (0..FSST_FRAGMENT_SIZE)
                    .map(|row| {
                        format!(
                            "https://api.example.com/v2/orders/{:08x}/status",
                            batch * FSST_FRAGMENT_SIZE + row
                        )
                    })
                    .collect();
                VarBinViewArray::from_iter(
                    strings.iter().map(|s| Some(s.as_str())),
                    DType::Utf8(Nullability::NonNullable),
                )
                .into_array()
            })
            .collect()
    }

    #[divan::bench]
    fn fsst_default(bencher: Bencher) {
        let fragments = make_string_fragments();
        let compressor = BtrBlocksCompressorBuilder::empty()
            .with_new_scheme(&FSSTScheme)
            .build();
        let total_rows = fragments.len() * FSST_FRAGMENT_SIZE;
        bencher
            .with_inputs(|| (&fragments, SESSION.create_execution_ctx()))
            .counter(ItemsCount::new(total_rows))
            .bench_refs(|(fragments, ctx)| {
                for fragment in fragments.iter() {
                    compressor.compress(fragment, ctx).unwrap();
                }
            });
    }

    #[divan::bench]
    fn fsst_pretrained(bencher: Bencher) {
        let fragments = make_string_fragments();
        let pretrained = Arc::new(fsst_train_compressor(
            &fragments[0]
                .clone()
                .execute::<VarBinViewArray>(&mut SESSION.create_execution_ctx())
                .unwrap(),
        ));
        let compressor = BtrBlocksCompressorBuilder::empty()
            .with_new_scheme_arc(Arc::new(FSSTSchemeWithPretrained::new(pretrained)))
            .build();
        let total_rows = fragments.len() * FSST_FRAGMENT_SIZE;
        bencher
            .with_inputs(|| (&fragments, SESSION.create_execution_ctx()))
            .counter(ItemsCount::new(total_rows))
            .bench_refs(|(fragments, ctx)| {
                for fragment in fragments.iter() {
                    compressor.compress(fragment, ctx).unwrap();
                }
            });
    }
}

fn main() {
    divan::main()
}
