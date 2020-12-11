import hydra
from omegaconf import OmegaConf

from nlp_tasks import Config


@hydra.main(config_path="conf", config_name="nlp_tasks/config")
def my_app(cfg: Config) -> None:
    print(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    my_app()
