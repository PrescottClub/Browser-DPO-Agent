# scripts/02_collect_preferences.py

import argparse
import os
import sys

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import random
from typing import List, Dict, Any, Tuple

from tqdm import tqdm

from src.agent.model import AgentModel
from src.environment.interface import EnvironmentInterface
from src.utils.config import load_config
from src.utils.reproducibility import set_seed, get_seed_from_config
from src.utils.checkpoint_manager import CheckpointManager
from src.utils.mlflow_logger import MLflowLogger


class PreferenceSelector:
    """
    智能偏好选择器，基于质量指标选择最佳偏好对。
    """

    def __init__(self):
        self.quality_metrics = ['confidence', 'step_count', 'completion_time', 'action_validity']

    def select_preference_pairs(self, success_completions: List[Dict], fail_completions: List[Dict]) -> Tuple[str, str]:
        """
        基于质量指标选择最佳偏好对。

        Args:
            success_completions: 成功的完成列表
            fail_completions: 失败的完成列表

        Returns:
            Tuple[str, str]: (chosen, rejected) 偏好对
        """
        if not success_completions or not fail_completions:
            return None, None

        # 选择最高质量的成功样本
        best_success = self._select_best_success(success_completions)

        # 选择最低质量的失败样本（对比度最大）
        worst_failure = self._select_worst_failure(fail_completions)

        return best_success["completion"], worst_failure["completion"]

    def _select_best_success(self, success_completions: List[Dict]) -> Dict:
        """选择最高质量的成功样本"""
        if len(success_completions) == 1:
            return success_completions[0]

        scored_completions = []
        for completion in success_completions:
            score = self._calculate_success_quality_score(completion)
            scored_completions.append((completion, score))

        # 按分数降序排序，选择最高分
        scored_completions.sort(key=lambda x: x[1], reverse=True)
        return scored_completions[0][0]

    def _select_worst_failure(self, fail_completions: List[Dict]) -> Dict:
        """选择最低质量的失败样本"""
        if len(fail_completions) == 1:
            return fail_completions[0]

        scored_completions = []
        for completion in fail_completions:
            score = self._calculate_failure_quality_score(completion)
            scored_completions.append((completion, score))

        # 按分数升序排序，选择最低分
        scored_completions.sort(key=lambda x: x[1])
        return scored_completions[0][0]

    def _calculate_success_quality_score(self, completion: Dict) -> float:
        """计算成功样本的质量分数"""
        score = 0.0
        completion_text = completion["completion"]

        # 1. 检查是否包含清晰的思考过程
        if "thought:" in completion_text.lower() or "thinking:" in completion_text.lower():
            score += 2.0

        # 2. 检查动作格式的正确性
        if "action:" in completion_text.lower():
            score += 2.0
            # 检查动作格式
            action_part = completion_text.split("action:")[-1].strip()
            if any(action in action_part.upper() for action in ["CLICK", "TYPE", "SELECT", "CHECK"]):
                score += 1.0

        # 3. 检查响应长度（适中的长度通常质量更好）
        response_length = len(completion_text.split())
        if 10 <= response_length <= 50:  # 适中长度
            score += 1.0
        elif response_length > 50:  # 过长可能包含冗余信息
            score -= 0.5

        # 4. 检查是否包含选择器
        if 'selector=' in completion_text:
            score += 1.0

        return score

    def _calculate_failure_quality_score(self, completion: Dict) -> float:
        """计算失败样本的质量分数（分数越低表示失败得越彻底）"""
        score = 0.0
        completion_text = completion["completion"]

        # 1. 如果包含明显的错误格式，分数更低
        if "action:" not in completion_text.lower():
            score -= 2.0

        # 2. 如果包含无效的动作
        if any(invalid in completion_text.lower() for invalid in ["invalid", "error", "fail", "cannot"]):
            score -= 1.0

        # 3. 如果响应过短或过长
        response_length = len(completion_text.split())
        if response_length < 5:  # 过短
            score -= 1.0
        elif response_length > 100:  # 过长且失败
            score -= 0.5

        # 4. 如果缺少选择器
        if 'selector=' not in completion_text:
            score -= 1.0

        return score


def main():
    print("--- 启动DPO偏好数据自动收流程 ---")

    # 0. 解析命令行参数
    parser = argparse.ArgumentParser(description="偏好数据收集脚本")
    parser.add_argument(
        "--config_path",
        type=str,
        default="config.yaml",
        help="配置文件路径 (默认: config.yaml)",
    )
    parser.add_argument(
        "--overwrite", 
        action="store_true",
        help="允许覆盖现有的偏好数据文件"
    )
    args = parser.parse_args()

    # 1. 加载配置
    config = load_config(args.config_path)

    # 2. 设置随机种子确保可复现性
    seed = get_seed_from_config(config)
    set_seed(seed)

    # 3. 使用深度MLflow集成
    with MLflowLogger("DPO-Driver Preferences", args.config_path) as mlflow_logger:
        print(f"--- 启动偏好数据收集流程 (MLflow Run ID: {mlflow_logger.get_run_id()}) ---")
        
        # 记录配置参数
        mlflow_logger.log_config_params(config)

        # 4. 从配置中获取参数
        model_name = config.model.base_model_name
        
        # 使用checkpoint管理器获取SFT checkpoint路径
        checkpoint_manager = CheckpointManager("./models")
        try:
            adapter_path = checkpoint_manager.get_checkpoint_path("sft")
            print(f"✓ 从checkpoint管理器获取SFT路径: {adapter_path}")
        except (FileNotFoundError, KeyError, ValueError) as e:
            print(f"[警告] 无法从checkpoint管理器获取SFT路径: {e}")
            print("回退到配置文件中的路径...")
            # 回退到配置文件路径加checkpoint-100（保持向后兼容）
            adapter_path = config.paths.sft_adapter_path + "/checkpoint-100"
            print(f"使用回退路径: {adapter_path}")
        
        # 生成安全的输出文件路径
        preference_dir = os.path.dirname(config.paths.preference_data)
        os.makedirs(preference_dir, exist_ok=True)
        
        safe_output_file = f"{config.paths.preference_data}_{mlflow_logger.get_run_id()}"
        
        print(f"[安全] 安全偏好数据路径: {safe_output_file}")

        # 使用配置中的评估任务和集数
        tasks_to_collect = config.evaluation.tasks
        episodes_per_task = config.evaluation.num_episodes_per_task

        # 5. 加载SFT Agent
        agent = AgentModel.from_sft_adapter(
            base_model_name=model_name, adapter_path=adapter_path
        )

        # 6. 收集原始交互数据
        interaction_log = []
        for task_id in tasks_to_collect:
            print(f"\n--- 正在为任务 {task_id} 收集数据 ---")
            env = None
            try:
                env = EnvironmentInterface(task_id=task_id)
                for i in tqdm(range(episodes_per_task), desc=f"Processing {task_id}"):
                    try:
                        obs, info = env.reset()
                        prompt = obs["utterance"]

                        # Agent生成完整的响应
                        completion = agent.generate_completion(prompt)

                        # 从响应中提取最后一个action
                        # 这是一个简化的提取逻辑，未来可以改进
                        last_action = None
                        if "action:" in completion:
                            try:
                                last_action = completion.split("action:")[-1].strip()
                            except (IndexError, AttributeError) as e:
                                print(f"[警告] 动作提取失败: {e}")
                                last_action = None
                            except Exception as e:
                                print(f"[错误] 动作提取时发生未预期错误: {e}")
                                last_action = None

                        if last_action:
                            _, reward, terminated, _, _ = env.step(last_action)
                            if terminated:
                                interaction_log.append(
                                    {"prompt": prompt, "completion": completion, "reward": reward}
                                )
                        else:  # 如果没有生成有效的action
                            interaction_log.append(
                                {
                                    "prompt": prompt,
                                    "completion": completion,
                                    "reward": -1.0,  # 给予惩罚
                                }
                            )
                    
                    except KeyboardInterrupt:
                        print(f"[中断] 用户中断了回合 {i} 的处理")
                        raise  # 重新抛出，让上层处理
                    except (EnvironmentError, RuntimeError) as env_error:
                        print(f"[环境错误] 处理回合 {i} 时环境错误: {env_error}")
                        # 记录环境错误但继续处理
                        interaction_log.append(
                            {
                                "prompt": f"Environment error in episode {i}",
                                "completion": f"Environment failed: {str(env_error)}",
                                "reward": -1.0,
                            }
                        )
                        continue
                    except Exception as episode_error:
                        print(f"[未知错误] 处理回合 {i} 时发生未预期错误: {episode_error}")
                        print(f"[调试] 错误类型: {type(episode_error).__name__}")
                        # 记录失败的回合但继续处理
                        interaction_log.append(
                            {
                                "prompt": f"Unexpected error in episode {i}",
                                "completion": f"Episode failed unexpectedly: {str(episode_error)}",
                                "reward": -1.0,
                            }
                        )
                        continue

            except KeyboardInterrupt:
                print(f"[中断] 用户中断了任务 {task_id} 的处理")
                raise  # 重新抛出，让上层处理
            except (ImportError, ModuleNotFoundError) as import_error:
                print(f"[导入错误] 任务 {task_id} 缺少必要的依赖: {import_error}")
                continue
            except (FileNotFoundError, PermissionError) as file_error:
                print(f"[文件错误] 任务 {task_id} 文件访问错误: {file_error}")
                continue
            except Exception as task_error:
                print(f"[未知错误] 处理任务 {task_id} 时发生未预期错误: {task_error}")
                print(f"[调试] 错误类型: {type(task_error).__name__}")
                continue
            
            finally:
                # 确保环境总是被关闭，即使发生异常
                if env is not None:
                    try:
                        env.close()
                        print(f"✓ 环境 {task_id} 已安全关闭")
                    except Exception as close_error:
                        print(f"[警告] 关闭环境时发生错误: {close_error}")
                        # 不抛出异常，继续处理下一个任务

        # 7. 处理日志，构建偏好对
        print("\n--- 正在处理交互日志，构建偏好数据集 ---")
        preferences = []
        prompts = set([log["prompt"] for log in interaction_log])

        # 初始化智能偏好选择器
        preference_selector = PreferenceSelector()

        for prompt in prompts:
            prompt_logs = [log for log in interaction_log if log["prompt"] == prompt]

            # 分离成功和失败的案例，保留完整的log信息用于质量评估
            success_logs = [log for log in prompt_logs if log["reward"] == 1.0]
            fail_logs = [log for log in prompt_logs if log["reward"] <= 0.0]

            # 如果同时有成功和失败的案例，则创建偏好对
            if success_logs and fail_logs:
                # 使用智能选择器选择最佳偏好对
                chosen, rejected = preference_selector.select_preference_pairs(success_logs, fail_logs)

                if chosen and rejected:
                    preferences.append(
                        {"prompt": prompt, "chosen": chosen, "rejected": rejected}
                    )
                    print(f"✓ 智能选择偏好对: {prompt[:50]}...")
                else:
                    # 如果智能选择失败，回退到随机选择
                    chosen = random.choice([log["completion"] for log in success_logs])
                    rejected = random.choice([log["completion"] for log in fail_logs])
                    preferences.append(
                        {"prompt": prompt, "chosen": chosen, "rejected": rejected}
                    )
                    print(f"⚠ 回退到随机选择: {prompt[:50]}...")

        # 7. 保存数据集
        print(f"成功构建了 {len(preferences)} 条偏好数据。")
        with open(safe_output_file, "w") as f:
            for item in preferences:
                f.write(json.dumps(item) + "\n")
        print(f"偏好数据集已保存至 {safe_output_file}")
        
        # 8. 记录完成信息
        mlflow_logger.log_stage_completion(
            stage_name="preference_collection",
            preferences_count=len(preferences),
            output_file=safe_output_file,
            adapter_path=adapter_path
        )


if __name__ == "__main__":
    main()
