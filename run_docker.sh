docker image rm don_word_sim:0.1
docker build -t=don_word_sim:0.1 .

mkdir -p ~/Orthographic-DNN/params
mkdir -p ~/Orthographic-DNN/data

docker run \
		-v ~/Orthographic-DNN/params:/main_dir/params \
		-v ~/Orthographic-DNN/data:/main_dir/data \
		-it \
		--shm-size 8G \
		--gpus all \
		--ipc=host \
		--rm \
		--runtime=nvidia \
		don_word_sim:0.1