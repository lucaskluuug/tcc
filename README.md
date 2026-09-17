# tcc-logo-detection

TCC: detecção automática de logotipos, UCS
Lucas Klug Arndt. Orientador: Prof. Dr. Andre Gustavo Adami

Experimento que mede o quanto o erro de localização afeta o erro de classificação em detectores de logotipos, usando Faster R-CNN na base FlickrLogos-32.

**Estado atual e resultados: [relatorio_progresso.md](relatorio_progresso.md)**

## Como funciona

Cada modelo é avaliado em dois fluxos sobre o conjunto de teste (P3):

- **Fluxo 1**: o modelo recebe a região já localizada pela anotação e só classifica a marca.
- **Fluxo 2**: o modelo recebe a imagem completa e faz tudo: localiza e classifica.

A diferença entre os dois é a medida do impacto do erro de localização.

## Estrutura

```
configs/      parâmetros de dados, modelo e métricas
scripts/      etapas do pipeline, numeradas na ordem de execução
src/logoloc/  implementação (dados, modelos, treino, avaliação)
results/      relatório gerado, gráficos e métricas por modelo
tests/        testes das métricas e da leitura do dataset
```

## Executando

A base FlickrLogos-32 precisa ser obtida com os autores e extraída em `data/raw/`.

```bash
pip install -r requirements.txt
python scripts/01_prepare_dataset.py
python scripts/04_train_faster_rcnn.py --model-config configs/faster_rcnn/resnet50.yaml --run-name frcnn_r50_v1
python scripts/06_evaluate_faster_rcnn.py --checkpoint outputs/runs/frcnn_r50_v1/model_last.pt --model-config configs/faster_rcnn/resnet50.yaml --run-name frcnn_r50_v1
python scripts/08_consolidate_results.py
python scripts/09_generate_report.py
```

O treino é feito no Google Colab pelos notebooks `logoloc_colab.ipynb` (treino) e `logoloc_colab_eval.ipynb` (avaliação e consolidação). O checkpoint é salvo a cada época, então uma sessão interrompida é retomada de onde parou.

```bash
pytest
```
