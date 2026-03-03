export WANDB_INIT_TIMEOUT='600'
export RAY_TMPDIR='/mnt/weka/aisg/.cache/ray_cache'
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export DS_ACCELERATOR=cuda
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export HYDRA_FULL_ERROR=1

unset ROCR_VISIBLE_DEVICES

# DATA_PATH=/mnt/weka/aisg/post_training_team/code_forge/verl_gemma/data/DeepMath-103k
DATA_PATH=/mnt/weka/aisg/users/raymond/data
# EXPERIMENT_NAME=ppo/gemma/deepmath-103k/reward_system_1
EXPERIMENT_NAME=ppo_gemma_instruction_following_if_reward
CUSTOM_REWARD_FUNCTION_PATH=/mnt/weka/aisg/post_training_team/code_forge/verl_gemma/verl/verl/utils/reward_score/reward_if.py
# CUSTOM_REWARD_FUNCTION_PATH=/mnt/weka/aisg/post_training_team/code_forge/verl_gemma/verl/verl/workers/reward_manager/dapo_test.py
CUSTOM_REWARD_FUNCTION_NAME=get_reward_score_batch
# CUSTOM_REWARD_MANAGER=reward_system
CUSTOM_REWARD_MANAGER=if_reward

MODEL_PATH=google/gemma-3-4b-it

cd /mnt/weka/aisg/post_training_team/code_forge/verl_gemma/verl/verl/utils/reward_score

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    custom_reward_function.path=$CUSTOM_REWARD_FUNCTION_PATH \
    custom_reward_function.name=$CUSTOM_REWARD_FUNCTION_NAME \
    reward_model.reward_manager=$CUSTOM_REWARD_MANAGER \
    data.train_files=$DATA_PATH/if_train.parquet \
    data.val_files=$DATA_PATH/if_test.parquet \
    data.train_batch_size=128 \
    data.max_prompt_length=1024 \
    data.max_response_length=4095 \
    data.truncation=left \
    data.return_multi_modal_inputs=False \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.load_format=auto \
    actor_rollout_ref.rollout.max_num_batched_tokens=5120 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.strategy=fsdp2 \
    critic.model.path=$MODEL_PATH \
    critic.ppo_micro_batch_size_per_gpu=4 \
    trainer.val_before_train=True \
    trainer.logger='["console","wandb"]'\
    trainer.project_name='rl_expts' \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=4 \
    trainer.critic_warmup=0 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.log_val_generations=5 \
    trainer.total_epochs=15 2>&1 | tee verl_demo.log