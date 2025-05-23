import os

# 定义项目结构
project_structure = {
    "configs": ["default.yaml", "model.yaml", "mmimdb.yaml"],
    "data": ["raw/", "processed/", "splits/"],
    "datasets": ["base_dataset.py", "multimodal_dataset.py", "transforms.py"],
    "models": [
        "__init__.py",
        "transformer.py",
        "quality_aware_prompting.py",
        "modality_fusion.py",
    ],
    "prompts": ["modality_prompt.py", "quality_prompt.py", "cross_modal_prompt.py"],
    "training": ["train.py", "eval.py", "losses.py", "optimizer.py"],
    "utils": ["logger.py", "my_metrics.py", "visualization.py", "helper.py"],
    "scripts": ["preprocess_data.py", "run_experiment.sh", "inference.py"],
    "results": ["logs/", "checkpoints/", "predictions/"],
}


# 创建文件和目录
def create_project_structure(base_path=""):
    for folder, files in project_structure.items():
        folder_path = os.path.join(base_path, folder)
        os.makedirs(folder_path, exist_ok=True)

        for file in files:
            file_path = os.path.join(folder_path, file)
            if file.endswith("/"):  # 创建子目录
                os.makedirs(file_path, exist_ok=True)
            else:  # 创建文件
                with open(file_path, "w", encoding="utf-8") as f:
                    if file.endswith(".py"):
                        f.write("# " + file.replace(".py", "").replace("_", " ").title() + "\n")
                    elif file.endswith(".yaml"):
                        f.write("# YAML 配置文件\n")

    # 创建 README.md 和 requirements.txt
    with open(os.path.join(base_path, "README.md"), "w", encoding="utf-8") as f:
        f.write("# Your Project\n\n")
        f.write("这是一个多模态 Transformer 研究项目。")

    with open(os.path.join(base_path, "requirements.txt"), "w", encoding="utf-8") as f:
        f.write("# 依赖库列表\n")

    print(f"✅ 项目结构已生成在 `{base_path}/` 下！")


# 运行脚本
create_project_structure()
