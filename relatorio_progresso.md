# TCC2: relatório de progresso

**Aluno:** Lucas Klug Arndt. **Orientador:** Prof. Dr. Andre Gustavo Adami, UCS
**Data:** setembro de 2026

---

## 1. Resumo

O pipeline completo do experimento está implementado e rodando. Treinei e avaliei o Faster R-CNN com dois backbones (ResNet-50 e ResNet-101) na FlickrLogos-32, usando a partição oficial (P1/P2/P3), e já tenho os dois fluxos de avaliação funcionando sobre o conjunto de teste.

Os resultados de detecção estão coerentes com a literatura, o que indica que o pipeline está correto. A análise dos números confirmou uma limitação do protocolo do Fluxo 1 que havia sido levantada na apresentação da etapa 1. Quantifiquei o problema, implementei uma correção e rodei o experimento corrigido nos dois modelos (seção 2). O resultado principal está na seção 5: uma curva de degradação da classificação em função do IoU da caixa, medida sobre as mesmas instâncias em todos os níveis.

---

## 2. O percurso metodológico

Esta seção registra como o método chegou ao formato atual, porque o caminho faz parte do resultado.

### 2.1 A pergunta

Quanto do erro de classificação de um detector vem de ter errado a localização? Em outras palavras: quando o modelo diz "Adidas" onde havia uma Nike, o quanto disso é culpa de ele estar olhando para a região errada?

Para responder, é preciso comparar o desempenho do modelo em duas condições: com a localização que ele mesmo produz, e com a localização perfeita. A diferença entre as duas seria o custo do erro de localização.

### 2.2 O método inicial: dois fluxos

Partimos de um método com dois fluxos, ambos avaliados sobre o mesmo conjunto de teste (P3, 1.602 instâncias anotadas):

- **Fluxo 2, localização do modelo.** O detector recebe a imagem completa e faz o trabalho inteiro, localizar e classificar. Acumula os dois tipos de erro.
- **Fluxo 1, localização ideal.** Fornecemos ao modelo a imagem recortada pela caixa anotada, esperando que, sem precisar localizar nada, o resultado medisse a capacidade de classificação isolada. Esse fluxo funcionaria como o teto de desempenho.

A métrica de interesse seria a diferença entre os dois: o quanto o modelo perde ao ter que localizar sozinho, em vez de receber a região pronta.

O recorte foi gerado com 10% de margem ao redor da caixa anotada e redimensionado para 640x640, a mesma resolução de entrada usada no treino. Geramos também uma variante sem margem, como análise de sensibilidade ao enquadramento.

### 2.3 O que os primeiros resultados mostraram

Treinamos o Faster R-CNN com dois backbones (ResNet-50 e ResNet-101) na partição oficial e avaliamos os dois fluxos.

O Fluxo 2 deu mAP@0,5 de 0,501 e 0,497, compatível com o baseline da literatura para treino desse tamanho, o que indicava que o pipeline estava correto. Mas o Fluxo 1 deu 0,383 e 0,554: o teto ficou **abaixo** da classificação que o próprio Fluxo 2 alcançava nos casos bem localizados (0,675 e 0,700). A métrica de impacto, definida como a diferença entre os dois, deu **negativa**: -0,502 e -0,337. Lido ao pé da letra, isso diria que errar a localização melhora a classificação.

### 2.4 O diagnóstico

O número absurdo não era um achado, era um sintoma. Investigando, encontramos três causas somadas:

1. **Falha de detecção contava como erro de classificação.** O recorte entra no detector como uma imagem nova, então a RPN ainda precisa encontrar o logotipo dentro dele. Quando não encontrava, o resultado era registrado como classe errada. Isso aconteceu em 16,9% e 17,3% dos recortes.

2. **Os dois fluxos mediam populações diferentes.** O Fluxo 1 cobria as 1.602 instâncias; a acurácia condicionada do Fluxo 2 cobria só as que o modelo já tinha localizado bem, os 68% a 71% mais fáceis. Comparar as duas era comparar denominadores diferentes.

3. **O recorte é uma imagem fora da distribuição de treino.** O modelo aprendeu a ver logotipos ocupando uma fração de uma cena; o recorte esticado para 640x640 põe o objeto preenchendo o quadro inteiro, sem contexto.

Corrigindo apenas as definições (causas 1 e 2):

| Modelo | Fluxo 1 como estava | Fluxo 1 só onde houve detecção | Fluxo 2 condicionado |
|---|---|---|---|
| ResNet-50 | 0,383 | 0,461 | 0,675 |
| ResNet-101 | 0,554 | 0,670 | 0,700 |

A diferença do ResNet-101 caía de 0,337 para 0,030. Quase tudo era artefato de definição de métrica, não achado sobre os modelos.

A causa 3, porém, permanecia. A evidência mais direta dela: a variante de recorte justo, sem a margem de 10%, derrubava a acurácia de 0,383 para 0,170 no ResNet-50 e de 0,554 para 0,314 no ResNet-101. Um recorte mais justo deveria, em tese, ser uma localização melhor. O desempenho cair pela metade mostra que o protocolo media sensibilidade ao enquadramento, não capacidade de classificar.

Essa limitação havia sido levantada na apresentação da etapa 1. Os dados confirmaram a ressalva e permitiram quantificá-la.

### 2.5 A mudança de abordagem

O problema de fundo era que o recorte **simulava a localização perfeita por fora do modelo**, e ao fazer isso criava uma tarefa diferente daquela para a qual o modelo foi treinado.

A correção foi passar a **fornecer a localização por dentro**: a imagem completa passa pelo backbone exatamente como no treino, e a caixa anotada é injetada direto no RoI pool, como se fosse a proposta da RPN. O objeto continua na cena, a escala é a mesma do treino, e toda caixa recebe uma distribuição de classes, então a categoria de não ter detectado nada deixa de existir.

E, como a caixa virou um parâmetro de entrada, foi possível **degradá-la de forma controlada**: perturbamos a caixa anotada até níveis-alvo de IoU (1,0 a 0,3), sorteando uma direção de deslocamento e escala e fazendo busca binária na magnitude até atingir o alvo, registrando o IoU efetivamente obtido. As mesmas 1.602 instâncias são medidas em todos os níveis.

Essa última parte muda o tipo de afirmação que o trabalho consegue fazer. A curva original era observacional: instâncias com IoU maior classificam melhor, o que pode refletir apenas a dificuldade da instância, já que logotipos grandes e nítidos são fáceis de localizar e de classificar ao mesmo tempo. Com a perturbação controlada, a instância é a mesma em todos os níveis e a única coisa que muda é a caixa. Sai uma relação de dose-resposta, não uma correlação.

As duas correções rodam juntas em `scripts/10_localization_sensitivity.py`. O nível de IoU 1,0 é o Fluxo 1 corrigido; os demais dão a curva de degradação. Nenhuma delas exigiu retreinar os modelos.

### 2.6 Uma limitação que virou achado

A injeção de caixas no RoI pool **só é possível por causa da arquitetura de dois estágios**. Em um detector de um estágio não existe etapa de proposta separável para interceptar: localização e classificação saem juntas de uma grade densa. Ou seja, a própria possibilidade de decompor o erro dessa forma depende da família do detector. Isso me parece um argumento mais concreto para a comparação entre famílias do que comparar mAP, e pretendo levar para a discussão.

### 2.7 Resumo do percurso

Começamos medindo o contrafactual por fora do modelo, o que criou uma tarefa diferente e produziu um número impossível. Passamos a fornecê-lo por dentro, o que além de corrigir a medição abriu a possibilidade de variar a localização de forma controlada e transformar uma correlação em relação causal. Os resultados dessa versão final estão nas seções 5 e 6.

---

## 3. O que foi implementado

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

## 4. Configuração experimental

**Dados:** FlickrLogos-32, partição oficial. P1 = 320 imagens de treino (10 por classe), expandidas para 1.280 com aumento de dados (espelhamento, escala e cor). P3 = 3.960 imagens de teste (960 com logotipo + 3.000 sem), totalizando 1.602 instâncias anotadas.

**Modelos:** Faster R-CNN com ResNet-50-FPN e ResNet-101-FPN, pesos pré-treinados, 30 épocas, lote 8, SGD com taxa de aprendizado 0,005, resolução de entrada 640, limiar de confiança 0,25, NMS 0,5, máximo de 100 detecções por imagem.

**Métricas:** limiar de IoU principal 0,5, com 0,75 reportado em paralelo.

**Hardware:** GPU Tesla T4 (Colab). O ResNet-50 levou 98 minutos nas 30 épocas. O ResNet-101 foi treinado em duas sessões porque a primeira caiu na época 16, e o tempo registrado no log (60 minutos) cobre apenas as 14 épocas da segunda sessão, não o treino inteiro.

---

## 5. Resultados

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

### Gráfico observacional

O gráfico `results/iou_vs_classification.png` mostra a taxa de acerto de classificação por faixa de IoU da caixa predita pelo modelo. Faixas com menos de 20 amostras foram omitidas.

### Sensibilidade ao erro de localização (experimento controlado)

Caixas anotadas perturbadas até níveis controlados de IoU e injetadas direto na RoI head. As mesmas 1.602 instâncias em todos os níveis. Média de **5 sementes** de perturbação, com o desvio padrão entre elas. Gráfico em `results/localization_sensitivity.png`.

| IoU da caixa | ResNet-50 | ResNet-101 | Caixas acima do limiar |
|---|---|---|---|
| 1,0 (anotada) | 0,592 | 0,604 | 51% |
| 0,9 | 0,587 ± 0,001 | 0,602 ± 0,002 | 51% |
| 0,8 | 0,583 ± 0,004 | 0,599 ± 0,005 | 49% |
| 0,7 | 0,573 ± 0,003 | 0,587 ± 0,006 | 44% |
| 0,6 | 0,551 ± 0,004 | 0,572 ± 0,003 | 35% |
| 0,5 | 0,525 ± 0,006 | 0,550 ± 0,005 | 17% |
| 0,4 | 0,440 ± 0,007 | 0,474 ± 0,004 | 2% |
| 0,3 | 0,335 ± 0,007 | 0,366 ± 0,009 | 1% |

O nível 1,0 não tem desvio porque a caixa anotada não é perturbada. A última coluna é a fração de caixas cujo score de classe supera o limiar de confiança do detector (0,25), ou seja, que virariam uma detecção emitida; os dois modelos ficam muito próximos e a coluna mostra a média.

**Sobre a robustez.** O desvio entre sementes fica sempre abaixo de 0,01, uma ordem de grandeza menor que os efeitos medidos. A queda de 1,0 para 0,5 é de 0,067 ± 0,006 no ResNet-50 e 0,054 ± 0,005 no ResNet-101; nenhuma semente individual chega perto de anular o efeito (o mínimo observado foi 0,047). A diferença entre os dois modelos também se sustenta: em 7 dos 8 níveis, a pior semente do ResNet-101 ainda supera a melhor do ResNet-50. E a semente 42, rodada duas vezes em sessões diferentes, reproduziu valores idênticos até a última casa, o que confirma que o protocolo é determinístico dada a semente.

---

## 6. Análise

**A hipótese central tem sustentação nos dados, agora de forma controlada.** O gráfico observacional já mostrava que caixas com IoU maior classificam melhor, mas essa correlação podia ser só dificuldade da instância (logotipos grandes e nítidos são fáceis de localizar e de classificar ao mesmo tempo). O experimento de sensibilidade elimina esse confundidor: são as mesmas 1.602 instâncias em todos os níveis, e a única coisa que muda é a caixa. A curva continua monotônica, nos dois backbones. O efeito é real.

**Os números batem com a literatura.** O mAP@0,5 de ~0,50 é compatível com o baseline de Faster R-CNN reportado por Su et al. (2017), de 50,4%, em um regime de treino de tamanho parecido (10 imagens por classe). Isso é um bom indicativo de que o pipeline não tem erro grosseiro. Vale registrar que a maioria dos trabalhos da área **não** usa a partição oficial da FlickrLogos-32: usam divisões próprias, bem maiores de treino, com mAP entre 50% e 90%. Essa é a razão de a comparação direta com a literatura ser limitada, e acho que isso merece um parágrafo no texto final.

**A classificação é robusta a erro moderado de localização, e degrada forte abaixo de IoU 0,5.** Entre a caixa anotada e IoU 0,7 a acurácia cai 1,8 a 2,0 pontos. Entre 1,0 e 0,5, cai 6,7 pontos no ResNet-50 e 5,4 no ResNet-101. Abaixo de 0,5 a queda acelera: 25,7 e 23,8 pontos até IoU 0,3. Esse é o custo do erro de localização sobre a classificação, medido nas mesmas instâncias, que é a pergunta do trabalho. O joelho da curva fica exatamente em IoU 0,5, e isso tem uma explicação no treino, não na métrica: `box_fg_iou_thresh = 0,5` é o limiar que o torchvision usa para decidir se uma proposta conta como objeto ou como fundo ao treinar a RoI head. Propostas abaixo de 0,5 foram apresentadas ao modelo como exemplos de fundo durante todo o treino. A curva cair justamente aí não é acaso: abaixo desse ponto, a rede está fazendo o que aprendeu a fazer, que é chamar aquilo de fundo.

**O erro de localização se manifesta mais como detecção perdida do que como marca errada.** A última coluna da tabela de sensibilidade explica a decomposição de erro: uma caixa em IoU 0,5 só passa do limiar de confiança 17% das vezes, e em 0,4 praticamente nunca (2%). Uma proposta mal localizada não vira uma predição com a marca errada; ela é suprimida pelo limiar e a instância acaba contada como "não detectada". Por isso o erro de localização puro aparece como só 5% da massa de erro enquanto a detecção perdida responde por metade. Note que a confiança desaba bem mais rápido que a acurácia: entre IoU 1,0 e 0,5 a acurácia cai 11% em termos relativos, enquanto a fração de caixas acima do limiar cai 67%. O modelo continua sabendo qual é a marca; ele deixa de ter confiança para dizer.

**O gargalo não é a RPN, é a confiança da cabeça de classificação.** Mesmo com a caixa anotada exata, só 51% das instâncias recebem confiança acima do limiar. Isso bate com a revocação do Fluxo 2 (0,48). Ou seja: se a RPN propusesse a caixa perfeita para todo logotipo, o detector ainda perderia metade. Entre as caixas que passam do limiar, a acurácia é de 0,88; entre as que não passam, 0,30. A cabeça sabe classificar quando está confiante; o problema é que ela é conservadora demais para metade dos logotipos verdadeiros, o que é esperado com 320 imagens de treino e uma proporção enorme de regiões de fundo nas amostras da RoI head. Na versão anterior deste relatório eu havia atribuído o gargalo à RPN; os dados corrigem isso.

**A vantagem do ResNet-101 na classificação era quase toda artefato do recorte.** Pelo protocolo antigo, o Fluxo 1 dava 0,383 contra 0,554, uma diferença de 17 pontos. Com a injeção, dá 0,592 contra 0,604: 1,2 ponto. O ResNet-50 não classificava pior; ele era mais sensível à distorção do recorte redimensionado. O que sobra de vantagem do backbone mais profundo é uma robustez maior a caixas ruins: a diferença entre os dois cresce de 1,2 ponto com a caixa anotada para 3,4 pontos em IoU 0,4, e a queda de 1,0 para 0,5 é menor no ResNet-101 (5,4 contra 6,7 pontos). Ou seja, a profundidade extra não ajuda a classificar melhor uma região bem delimitada, ajuda a tolerar uma região mal delimitada. O mAP praticamente igual entre os dois (0,501 e 0,497) fica coerente com isso: na prática o detector opera com caixas boas, onde a vantagem é mínima.

---

## 7. Escopo

Conforme conversado e sugerido na apresentação da etapa 1, o **YOLOv8 saiu do escopo** e o trabalho está centrado no Faster R-CNN, comparando as duas profundidades de backbone. Antes de consolidar essa decisão cheguei a treinar e avaliar um YOLOv8s; os números estão no anexo (`results/relatorio.md`) apenas como registro do que foi executado, não como parte da análise.

---

## 8. Próximos passos

1. Escrever a seção de resultados em torno da curva de sensibilidade, com a decomposição de erro e o gráfico observacional como apoio.
2. Rever o posicionamento da contribuição em relação a trabalhos de decomposição de erro em detecção (TIDE, Bolya et al., 2020; Hoiem et al., 2012), que já fazem a separação localização/classificação em detecção genérica. A parte original aqui é a medição do contrafactual de localização perfeita, não a decomposição em si. Acho importante deixar isso claro na introdução.
3. Escrever a seção de metodologia com os parâmetros já fixados e registrados, incluindo a descrição do protocolo de perturbação e injeção.

**Dúvidas para a orientação:** o enquadramento da contribuição no item 2 faz sentido? E o achado de que o gargalo está na confiança da cabeça de classificação (seção 6) merece um experimento próprio, por exemplo variando o limiar de confiança, ou fica como discussão?

---

## 9. Onde está cada coisa

- Código: repositório no GitHub
- Relatório gerado automaticamente com todas as métricas: `results/relatorio.md`
- Gráfico IoU × classificação: `results/iou_vs_classification.png`
- Curva de degradação por nível de IoU, os dois modelos juntos: `results/localization_sensitivity.png`
- Curva e tabela por modelo: `results/<modelo>/localization_sensitivity.png` e `.csv`
- Matrizes de confusão e relatórios por classe: `results/<modelo>/`
- Configurações de cada execução: `configs/`
