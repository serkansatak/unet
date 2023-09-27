from src import Config, ModelTrainer, BaseDataset, UNet
import prettyprinter as pp


def main():
    config = Config()

    pp.pprint(config.to_dict())

    if config.general.mode == "train":
        model_trainer = ModelTrainer(config)
        model_trainer.train()


if __name__ == "__main__":
    main()
