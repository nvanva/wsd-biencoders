import hydra
from omegaconf import DictConfig
from bemwsd_api import BEMWSD
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/wsd', methods=['POST'])
def wsd():
    data = request.get_json()
    res = bemwsd.wsd(data['usages'], data['inventory'])
    return res


@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    global bemwsd
    bemwsd = BEMWSD(cfg)

    app.run(
        host=cfg.server.host,
        port=cfg.server.port,
        debug=cfg.server.debug,
        use_reloader=False
    )


if __name__ == "__main__":
    main()

