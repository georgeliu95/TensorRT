for wtype in dense sparse; do
    for prec in int8-qat; do
        for bs in 1 8 32 128 256 1024; do
            for slen in 128 384; do
                FNAME="megatron_large_b${bs}_s${slen}_${wtype}_${prec}"

                echo "Building engine for Megatron-large w/ slen $slen and $wtype,$prec weights"
                CKPT_PATH=/workspace/models/bert/megatron/demobert/${wtype}_${prec}
                BUILD_ARGS="--fp16 --strict"
                if [ "$prec" = "int8-qat" ]; then
                    BUILD_ARGS="--int8 -il ${BUILD_ARGS}"
                fi
                if [ "$wtype" = "sparse" ]; then
                    BUILD_ARGS="${BUILD_ARGS} -sp"
                fi
                BUILD_ARGS="-c models/fine-tuned/bert_tf_ckpt_large_qa_squad2_amp_${slen}_v19.03.1 -b ${bs} -s ${slen} -o engines/${FNAME}.engine --megatron --pickle $CKPT_PATH/model.pkl ${BUILD_ARGS} -v models/fine-tuned/bert_tf_ckpt_large_qa_squad2_amp_${slen}_v19.03.1/vocab.txt"
                echo "Running builder_varseqlen.py ${BUILD_ARGS}"
                if [ ! -f engines/${FNAME}.engine ]; then
                    python3 builder_varseqlen.py ${BUILD_ARGS}
		    sleep 3;
                fi
            done
        done
    done
done

for wtype in dense sparse; do
    for prec in int8-qat; do
        for bs in 1 8 32 128 256 1024; do
            for slen in 128 384; do
                FNAME="megatron_large_b${bs}_s${slen}_${wtype}_${prec}"
                echo "" > log_${FNAME}.txt

                echo "Running inference_varseqlen.py ${INFER_ARGS}"
                INFER_ARGS="-e engines/${FNAME}.engine -s ${slen} -sq ./squad/dev-v1.1.json -v models/fine-tuned/bert_tf_ckpt_large_qa_squad2_amp_${slen}_v19.03.1/vocab.txt -o ./predictions_${FNAME}.json"
                if [ "$slen" = "128" ]; then
                    INFER_ARGS="${INFER_ARGS} --doc-stride 32"
                fi
                if [ ! -f ./predictions_${FNAME}.json ]; then
                    python3 inference_varseqlen.py ${INFER_ARGS} | tee -a log_${FNAME}.txt
                fi
                python3 squad/evaluate-v1.1.py  squad/dev-v1.1.json  ./predictions_${FNAME}.json 90 2>&1 | tee -a log_${FNAME}.txt
                PERF_ARGS="-e engines/${FNAME}.engine -b ${bs} -s ${slen} -w 100 -i 2000 --enable_graph"
                build/perf ${PERF_ARGS} 2>&1 | tee -a log_${FNAME}.txt
            done
        done
    done
done
