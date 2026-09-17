# TCC2: relatório de progresso

**Aluno:** Lucas Klug Arndt. **Orientador:** Prof. Dr. Andre Gustavo Adami, UCS
**Data:** setembro de 2026

---

## 1. Resumo

O pipeline completo do experimento está implementado e rodando. Treinei e avaliei o Faster R-CNN com dois backbones (ResNet-50 e ResNet-101) na FlickrLogos-32, usando a partição oficial (P1/P2/P3), e já tenho os dois fluxos de avaliação funcionando sobre o conjunto de teste.

Os resultados de detecção estão coerentes com a literatura, o que indica que o pipeline está correto. A análise dos números, porém, confirmou uma limitação do protocolo do Fluxo 1 que havia sido levantada na apresentação da etapa 1. Quantifiquei o problema e proponho uma correção na seção 6.

---

## 2. O que foi implementado

| Etapa | Script | O que faz |
|---|---|---|
| Preparação | `01_prepare_dataset.py` | Lê a FlickrLogos-32, valida as partições oficiais, aplica aumento de dados em P1 e gera os recortes do Fluxo 1 |
| Estimativa | `03_estimate_cost.py` | Estima tempo de treino por configuração antes de gastar GPU |
| Treino | `04_train_faster_rcnn.py` | Treina o Faster R-CNN, salvando checkpoint a cada época |
| Avaliação | `06_evaluate_faster_rcnn.py` | Roda os dois fluxos sobre P3 e calcula todas as métricas |
| Consolidação | `08_consolidate_results.py` | Monta a tabela comparativa e o gráfico IoU × classificação |
| Relatório | `09_generate_report.py` | Gera o relatório consolidado com todos os modelos |
| Sensibilidade | `10_localization_sensitivity.py` | Perturba as caixas anotadas em níveis de IoU e mede a classificação em cada nível |

As métricas (IoU, precisão, revocação, F1, AP, mAP e a decomposição de erro) foram implementadas do zero, sem usar biblioteca pronta, para ter controle sobre o cálculo.

O pipeline suporta interrupção e retomada: o checkpoint guarda pesos, otimizador e agendador de taxa de aprendizado, e o treino continua da última época concluída. Isso foi necessário porque as sessões do Colab caem sem aviso. O ResNet-101 acabou sendo treinado em duas sessões por causa disso.

---

## 3. Configuração experimental

**Dados:** FlickrLogos-32, partição oficial. P1 = 320 imagens de treino (10 por classe), expandidas para 1.280 com aumento de dados (espelhamento, escala e cor). P3 = 3.960 imagens de teste (960 com logotipo + 3.000 sem), totalizando 1.602 instâncias anotadas.

**Modelos:** Faster R-CNN com ResNet-50-FPN e ResNet-101-FPN, pesos pré-treinados, 30 épocas, lote 8, SGD com taxa de aprendizado 0,005, resolução de entrada 640, limiar de confiança 0,25, NMS 0,5, máximo de 100 detecções por imagem.

**Métricas:** limiar de IoU principal 0,5, com 0,75 reportado em paralelo.

**Hardware:** GPU Tesla T4 (Colab). O ResNet-50 levou 98 minutos nas 30 épocas. O ResNet-101 foi treinado em duas sessões porque a primeira caiu na época 16, e o tempo registrado no log (60 minutos) cobre apenas as 14 épocas da segunda sessão, não o treino inteiro.

---

## 4. Resultados

### Tabela comparativa

| Modelo | Fluxo 1 (acurácia) | Fluxo 1 sem padding | Fluxo 2 mAP@0,5 | mAP@0,75 | Precisão | Revocação | IoU médio (VP) |
|---|---|---|---|---|---|---|---|
| Faster R-CNN ResNet-50 | 0,383 | 0,170 | 0,501 | 0,319 | 0,467 | 0,481 | 0,786 |
| Faster R-CNN ResNet-101 | 0,554 | 0,314 | 0,497 | 0,341 | 0,507 | 0,475 | 0,794 |

### Decomposição do erro (Fluxo 2, IoU 0,5)

| Modelo | Acertos | Erro de classificação | Erro de localização | Ambos | Falso positivo de fundo | Detecção perdida |
|---|---|---|---|---|---|---|
| ResNet-50 | 771 | 371 | 101 | 165 | 220 | 831 |
| ResNet-101 | 761 | 326 | 81 | 139 | 175 | 841 |

### Gráfico

O gráfico `results/iou_vs_classification.png` mostra a taxa de acerto de classificação por faixa de IoU da caixa predita.

---

## 5. Análise

**A hipótese central tem sustentação nos dados.** O gráfico IoU × classificação é monotônico: quanto maior a sobreposição da caixa predita com a anotação, maior a taxa de acerto da marca. O comportamento se repete nos dois backbones, o que indica que não é ruído de um modelo específico. Esse é o resultado mais sólido obtido até agora.

**Os números batem com a literatura.** O mAP@0,5 de ~0,50 é compatível com o baseline de Faster R-CNN reportado por Su et al. (2017), de 50,4%, em um regime de treino de tamanho parecido (10 imagens por classe). Isso é um bom indicativo de que o pipeline não tem erro grosseiro. Vale registrar que a maioria dos trabalhos da área **não** usa a partição oficial da FlickrLogos-32: usam divisões próprias, bem maiores de treino, com mAP entre 50% e 90%. Essa é a razão de a comparação direta com a literatura ser limitada, e acho que isso merece um parágrafo no texto final.

**Mais profundidade melhora a classificação, mas não a detecção.** O ResNet-101 melhorou bastante o Fluxo 1 (0,383 → 0,554) mas o mAP ficou praticamente igual (0,501 → 0,497), e a taxa de detecção perdida até subiu um pouco. A leitura que faço: a capacidade extra do backbone ajuda a cabeça de classificação a distinguir marcas, mas não resolve o gargalo do experimento, que é a RPN propor as regiões certas. Com 320 imagens de treino, o que limita não é reconhecer a marca, e sim achar o logotipo na cena.

**O maior erro não é nem localização nem classificação, é não detectar.** A detecção perdida responde por 48,6% (ResNet-50) e 53,2% (ResNet-101) de toda a massa de erro. Erro de localização puro é só 5,9% e 5,1%. Ou seja: a pergunta central do trabalho trata de uma fatia relativamente pequena do que de fato dá errado nesse regime de poucos dados. Não invalida o experimento, mas é um ponto que pretendo colocar de forma explícita na discussão.

---

## 6. Revisão do protocolo do Fluxo 1

Na apresentação da etapa 1 foi levantado que recortar a imagem pela caixa anotada pode não representar bem uma "localização perfeita". Os dados confirmaram isso, e consegui quantificar o problema.

**O que encontrei.** A métrica que compara os dois fluxos (`impacto_erro_localizacao`) dá negativa nos dois modelos, o que sugeriria, de forma absurda, que errar a localização *melhora* a classificação. Investigando, são dois efeitos somados:

1. **O Fluxo 1 penaliza falha de detecção como se fosse erro de classificação.** Quando o modelo não emite nenhuma predição sobre o recorte, isso entra como erro na acurácia. E acontece bastante: 16,9% dos recortes no ResNet-50 e 17,3% no ResNet-101. Isso não é erro de classificação, é o detector não reconhecendo aquela entrada.

2. **Os dois fluxos são medidos sobre populações diferentes.** O Fluxo 1 roda sobre todas as 1.602 instâncias. A acurácia condicionada do Fluxo 2 só olha as instâncias que o modelo já encontrou bem (confiança acima do limiar e IoU de pelo menos 0,5), ou seja, os 68% a 71% de casos mais fáceis.

Corrigindo os dois lados para a mesma base:

| Modelo | Fluxo 1 (como está) | Fluxo 1 só onde houve detecção | Fluxo 2 condicionado |
|---|---|---|---|
| ResNet-50 | 0,383 | 0,461 | 0,675 |
| ResNet-101 | 0,554 | 0,670 | 0,700 |

A diferença de 0,337 do ResNet-101 cai para 0,030. Ou seja: boa parte do resultado estranho era artefato da definição da métrica, não um achado sobre os modelos.

**Evidência adicional.** A variante de recorte justo (sem os 10% de margem) derruba a acurácia de 0,383 para 0,170 no ResNet-50 e de 0,554 para 0,314 no ResNet-101. Um recorte mais justo deveria, em tese, ser uma localização *melhor*. O desempenho cair pela metade mostra que o que está sendo medido depende fortemente do enquadramento, e não só da capacidade de classificar.

**Correção, já implementada e pronta para rodar** (nenhuma das duas exige retreinar):

1. **Experimento de perturbação controlada.** Em vez de recortar, perturbo as caixas anotadas até níveis-alvo de IoU (1,0; 0,9; 0,8; 0,7; 0,6; 0,5; 0,4; 0,3) e meço a classificação em cada nível. Como são as mesmas instâncias em todos os níveis, isso elimina o viés de dificuldade que contamina a curva atual, e transforma "IoU e acerto estão correlacionados" em "degradar a caixa em X custa Y de acurácia", que é exatamente a pergunta do trabalho. A perturbação sorteia uma direção (deslocamento e escala) e faz busca binária na magnitude até atingir o IoU desejado, registrando o IoU efetivamente obtido em vez de assumir o alvo.

2. **Injeção das caixas anotadas direto na cabeça de classificação.** No Faster R-CNN passo as caixas do ground truth como propostas para a RoI head, pulando a RPN. Isso mede classificação com localização perfeita sem tirar o objeto da cena, que é o problema do recorte atual. Também resolve a distorção do item 1 da lista anterior: como toda caixa recebe uma distribuição de classes, não existe mais "não detectou nada" sendo contado como erro de classificação.

As duas rodam juntas em `scripts/10_localization_sensitivity.py`. O nível de IoU 1,0 desse experimento é o Fluxo 1 corrigido, e os demais níveis dão a curva de degradação.

Um ponto que achei interessante: essa injeção **só é possível por causa da arquitetura de dois estágios**. Em um detector de um estágio não existe etapa de proposta separável para interceptar. Isso quer dizer que a própria possibilidade de decompor o erro depende da família do detector, o que me parece um achado relevante para a discussão.

---

## 7. Escopo

Conforme conversado e sugerido na apresentação da etapa 1, o **YOLOv8 saiu do escopo** e o trabalho está centrado no Faster R-CNN, comparando as duas profundidades de backbone. Antes de consolidar essa decisão cheguei a treinar e avaliar um YOLOv8s; os números estão no anexo (`results/relatorio.md`) apenas como registro do que foi executado, não como parte da análise.

---

## 8. Próximos passos

1. Rodar o experimento da seção 6 nos dois backbones e incorporar a curva de degradação aos resultados. O código está pronto e testado, falta executar na GPU.
2. Rever o posicionamento da contribuição em relação a trabalhos de decomposição de erro em detecção (TIDE, Bolya et al., 2020; Hoiem et al., 2012), que já fazem a separação localização/classificação em detecção genérica. A parte original aqui é a medição do contrafactual de localização perfeita, não a decomposição em si. Acho importante deixar isso claro na introdução.
3. Escrever a seção de metodologia com os parâmetros já fixados e registrados.

**Dúvida para a orientação:** o enquadramento da contribuição no item 2 faz sentido, ou é melhor centrar o texto em outro aspecto?

---

## 9. Onde está cada coisa

- Código: repositório no GitHub
- Relatório gerado automaticamente com todas as métricas: `results/relatorio.md`
- Gráfico IoU × classificação: `results/iou_vs_classification.png`
- Curva de degradação por nível de IoU: `results/<modelo>/localization_sensitivity.png` e `.csv`
- Matrizes de confusão e relatórios por classe: `results/<modelo>/`
- Configurações de cada execução: `configs/`
