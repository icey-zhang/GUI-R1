MODEL_PATH=checkpoints/Qwen2.5-VL-3B-R1-2
DATA_DIR=datasets/GUI-R1

python inference/inference_vllm_android.py --model_path ${MODEL_PATH} --data_path ${DATA_DIR}/androidcontrol_high_test.parquet
python inference/inference_vllm_android.py --model_path ${MODEL_PATH} --data_path ${DATA_DIR}/androidcontrol_low_test.parquet
python inference/inference_vllm_guiact_web.py --model_path ${MODEL_PATH} --data_path ${DATA_DIR}/guiact_web_test.parquet
python inference/inference_vllm_guiodyssey.py --model_path ${MODEL_PATH} --data_path ${DATA_DIR}/guiodyssey_test.parquet
python inference/inference_vllm_omniact_desktop.py --model_path ${MODEL_PATH} --data_path ${DATA_DIR}/omniact_desktop_test.parquet
python inference/inference_vllm_omniact_web.py --model_path ${MODEL_PATH} --data_path ${DATA_DIR}/omniact_web_test.parquet
python inference/inference_vllm_screenspot.py --model_path ${MODEL_PATH} --data_path ${DATA_DIR}/screenspot_test.parquet
python inference/inference_vllm_screenspot.py --model_path ${MODEL_PATH} --data_path ${DATA_DIR}/screenspot_pro_test.parquet