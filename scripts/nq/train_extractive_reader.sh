python train_extractive_reader.py \
	encoder.sequence_length=350 \
	train_files={path to the retriever train set results file} \
	dev_files={path to the retriever dev set results file}  \
	gold_passages_src={path to data.gold_passages_info.nq_train file} \
	gold_passages_src_dev={path to data.gold_passages_info.nq_dev file} \
	output_dir={path to output dir}