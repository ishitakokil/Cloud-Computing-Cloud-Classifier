import argparse
import datetime
import logging.config
from pathlib import Path
import yaml

import src.acquire_data as ad
import src.analysis as eda
import src.aws_utils as aws
import src.create_dataset as cd
import src.evaluate_performance as ep
import src.generate_features as gf
import src.score_model as sm
import src.train_model as tm

# Configure logger
logging.config.fileConfig("config/logging/local.conf")
logger = logging.getLogger("clouds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run full cloud classification pipeline"
    )
    parser.add_argument(
        "--config",
        default="config/default-config.yaml",
        help="Path to configuration file",
    )
    args = parser.parse_args()

    # Load configuration
    with open(args.config, "r") as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            logger.error("Failed to load YAML: %s", e)
            raise

    logger.info("Loaded config from %s", args.config)

    run_config = config["run_config"]
    paths = config["paths"]

    # Create artifacts directory
    timestamp = int(datetime.datetime.now().timestamp())
    artifacts_dir = Path("runs") / f"{run_config['name']}_{timestamp}"
    artifacts_dir.mkdir(parents=True)

    # Save config snapshot
    with open(artifacts_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)

    # Step 1: Acquire data
    ad.acquire_data(url=config["data_source"]["url"], save_path=Path(paths["raw_data"]))

    # Step 2: Create structured dataset
    data = cd.create_dataset(Path(paths["raw_data"]), config["data_source"])
    cd.save_dataset(data, Path(paths["cleaned_data"]))

    # Step 3: Feature generation
    features = gf.generate_features(data, config["generate_features"])
    cd.save_dataset(features, Path(paths["features_data"]))

    features = gf.generate_labels(
        features, method=config["labeling"]["method"], config=config["labeling"]
    )

    features = features.drop(columns=["IR_mean"])
    # Step 4: EDA figures (optional)
    figures_dir = artifacts_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    eda.save_figures(features, figures_dir)

    # Step 5: Model training
    model, train_df, test_df = tm.train_model(features, config["model"])
    tm.save_data(train_df, test_df, artifacts_dir)
    tm.save_model(model, Path(paths["model_output"]))

    # Step 6: Score model
    scores = sm.score_model(test_df, model, config["model"])
    sm.save_scores(scores, artifacts_dir / "scores.csv")

    # Step 7: Evaluate performance
    metrics = ep.evaluate_performance(scores, config["evaluation"])
    ep.save_metrics(metrics, Path(paths["metrics_output"]))

    # Step 8: Optional chart generation
    if config["evaluation"].get("plot_roc", False):
        ep.plot_roc_curve(scores, Path(paths["chart_output"]))

    # Step 9: Upload to S3
    if config["aws"].get("upload", False):
        aws.upload_artifacts(artifacts_dir, config["aws"])
        aws.upload_artifacts(Path("models"), config["aws"])
