# Manual — IA Científica para Análise de Difração de Raios X

Este texto é o guia prático para **alunos** e para o **ministrante**. O conteúdo científico está no notebook e nos slides; aqui está o *como usar*.

Material:

| O quê | Onde |
|---|---|
| Aula prática | [`notebooks/Minicurso_IA_Cientifica_DRX.ipynb`](../notebooks/Minicurso_IA_Cientifica_DRX.ipynb) |
| Slides (por módulo) | [`slides/index.html`](../slides/index.html) |
| Apresentação (abrir no navegador) | [link direto](https://raw.githack.com/220719/semana-da-fisica-2025/main/slides/apresentacao.html) |
| Difratogramas | [`data/Nd_10.csv`](../data/Nd_10.csv) … `Nd_50.csv` |

Não é necessário instalar Python no computador. O caminho oficial da turma é o **Google Colab**.

---

## 1. Conta Google

1. Use um e-mail Google (pessoal ou institucional).
2. Abra [colab.research.google.com](https://colab.research.google.com) e aceite os termos, se for a primeira vez.
3. CPU basta. GPU **não** acelera este pipeline (não há rede neural pesada).

---

## 2. Clonar o repositório no GitHub (visão geral)

O material público está em:

**https://github.com/220719/semana-da-fisica-2025**

Três formas de chegar no código:

### A — Abrir o notebook direto no Colab (mais simples)

1. Entre no repositório.
2. Clique no arquivo `notebooks/Minicurso_IA_Cientifica_DRX.ipynb`.
3. Use o badge **Open in Colab** do `README.md`, **ou** no Colab: *Arquivo → Abrir notebook → GitHub* e cole a URL do `.ipynb`.

Os CSV são baixados sozinhos da pasta `data/` no GitHub. Não precisa montar Drive.

### B — Clonar no Colab (repo inteiro: slides, dados, manual)

Crie um notebook vazio no Colab e rode:

```python
!git clone --depth 1 https://github.com/220719/semana-da-fisica-2025.git
%cd semana-da-fisica-2025
```

Depois:

```python
from google.colab import files
# ou abra pelo menu: Arquivo → Abrir notebook → pasta local da sessão
```

Na prática, o mais confortável depois do clone é:

*Arquivo → Abrir notebook → Upload* do `Minicurso_IA_Cientifica_DRX.ipynb` **já dentro** da pasta clonada, **ou** executar:

```python
%cd /content/semana-da-fisica-2025
```

e abrir o `.ipynb` a partir do navegador de arquivos do Colab (ícone de pasta à esquerda).

### C — Clone local (quem já usa Git)

```bash
git clone https://github.com/220719/semana-da-fisica-2025.git
cd semana-da-fisica-2025
```

Para rodar **fora** do Colab (opcional):

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebooks/Minicurso_IA_Cientifica_DRX.ipynb
```

A turma **não** precisa deste caminho C.

---

## 3. Google Colab — operação no dia da aula

1. Ambiente de execução → **Alterar o tipo de ambiente** → CPU → Salvar.
2. Execute em ordem: *Ambiente de execução → Executar tudo* (`Ctrl+F9` / `Cmd+F9`).
3. Se uma célula falhar, **não pule**: as seguintes dependem das variáveis anteriores (`patterns`, `processed`, `features`…).
4. Figuras também vão para a pasta `figuras/` da sessão (ícone de pasta à esquerda → download dos PNG).
5. A sessão do Colab **expira** (ociosidade ~90 min). O trabalho não fica salvo no GitHub automaticamente. Use *Arquivo → Salvar uma cópia no Drive* se quiser levar o notebook anotado para casa.

### Problemas frequentes

| Sintoma | O que fazer |
|---|---|
| `FileNotFoundError` em `data/` | A célula de leitura já cai na URL do GitHub. Confira internet. |
| `ModuleNotFoundError: google` | Só ocorre no copiloto se você pediu Gemini sem instalar. Rode `%pip install -q google-genai` **ou** ignore: o relatório local funciona sem isso. |
| Gráfico em branco | Rode de novo a célula; evite `plt.close('all')` no meio da aula. |
| “RAM esgotada” | Não deve acontecer neste curso. Reinicie o ambiente e execute tudo de novo. |

---

## 4. Dados de DRX

- Cinco arquivos: `Nd_10.csv` … `Nd_50.csv`.
- Duas colunas **sem cabeçalho**: `2θ` (graus), intensidade (u.a.).
- O notebook usa `header=None`. Não abra o CSV no Excel e salve de novo com cabeçalho, senão a leitura quebra.

Não envie os CSV para um LLM “para ele analisar o espectro”. O pipeline Python já resume os números; o LLM só lê esse resumo.

---

## 5. Slides (Reveal.js)

**Não abra o `.html` na página do GitHub** (`github.com/.../blob/main/slides/...`). Isso mostra o *código-fonte*, não a apresentação.

Use um destes caminhos:

1. **Link que já funciona no navegador** (recomendado para a turma):
   - [Apresentação completa](https://raw.githack.com/220719/semana-da-fisica-2025/main/slides/apresentacao.html)
   - [Índice por módulo](https://raw.githack.com/220719/semana-da-fisica-2025/main/slides/index.html)
2. **No seu PC**, depois do clone: clique duas vezes em `slides/apresentacao.html` (precisa de internet para o Reveal.js).
3. GitHub Pages (`220719.github.io/...`) só funciona depois de ligar Pages em Settings; o workflow ainda falha até isso estar ativo.

Teclado: setas, `F` tela cheia, `Esc` visão geral.

---

## 6. LLM copiloto (opcional)

O Módulo 6 funciona **sem chave**: gera um relatório local com os números do pipeline.

Para usar Gemini:

1. Crie uma chave em [Google AI Studio](https://aistudio.google.com/apikey) (conta Google).
2. No Colab, ícone de **chave** (Secrets) à esquerda.
3. Nome: `GEMINI_API_KEY` (exatamente assim).
4. Cole a chave e marque **Notebook access**.
5. Rode de novo a célula do copiloto.

Alternativa: `OPENAI_API_KEY` (conta OpenAI, uso pago na maior parte dos casos). O notebook tenta Gemini primeiro.

**Boas práticas em sala**

- Não cole a chave no notebook nem no chat.
- Não compartilhe prints com a chave visível.
- Trate o texto do LLM como rascunho: confronte com a tabela de métricas.

---

## 7. O que cada biblioteca faz

Já vêm no Colab:

| Pacote | Uso neste curso |
|---|---|
| `pandas` | Ler CSV e montar a tabela de features |
| `numpy` | Vetores, integral trapezoidal, Scherrer |
| `matplotlib` | Todos os gráficos |
| `scipy` | Savitzky–Golay, `find_peaks`, linha de base esparsa |
| `scikit-learn` | Ridge, floresta, LOO, R² / MAE / RMSE |

Opcionais (só copiloto em nuvem): `google-genai` ou `openai`.

Arquivo [`requirements.txt`](../requirements.txt) replica isso para quem roda local.

---

## 8. Regenerar notebook e slides (ministrante)

Na raiz do repositório:

```bash
python tools/make_colab_notebook.py
python tools/make_slides.py
```

Não edite o `.ipynb` “no braço” se pretende manter o gerador como fonte. Altere `tools/make_colab_notebook.py` e regenere.

---

## 9. Roteiro sugerido de 3 horas

| Tempo | Bloco |
|---|---|
| 0:00–0:20 | Módulo 1 (slides) + abrir o Colab juntos |
| 0:20–0:45 | Módulo 2 — leitura e figuras |
| 0:45–1:10 | Módulo 3 — pré-processamento |
| 1:10–1:40 | Módulo 4 — picos, FWHM, Scherrer |
| 1:40–2:20 | Módulo 5 — LOO e overfitting |
| 2:20–2:45 | Módulo 6 — relatório (com ou sem API) |
| 2:45–3:00 | Módulo 7 — `run_hybrid_pipeline` e fechamento |

---

## 10. Integridade científica (lembrete para a turma)

1. Números vêm das células de cálculo.
2. O LLM não é instrumento de medida.
3. Cinco amostras não sustentam um modelo publicável.
4. Dados sintéticos não substituem o difratômetro.
