from utils import Config, ModelTrainer, BaseDataset, UNet


def main():
    config = Config()

    if config.general.mode == "train":
        model_trainer = ModelTrainer(config)
        model_trainer.train()


if __name__ == "__main__":
    main()
