"""Gera slides HTML (Reveal.js via CDN) — um deck por módulo + índice + apresentação completa."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "slides"
ASSETS = ROOT / "assets"

CSS = """
:root { --uem: #0b3d5c; --accent: #c45c26; --ink: #1a1a1a; }
.reveal { font-family: "Segoe UI", "Liberation Sans", sans-serif; }
.reveal h1, .reveal h2, .reveal h3 { color: var(--uem); font-weight: 700; text-transform: none; }
.reveal h1 { font-size: 1.85em; }
.reveal h2 { font-size: 1.35em; }
.reveal .subtitle { color: var(--accent); font-size: 0.7em; font-weight: 600; letter-spacing: 0.04em; }
.reveal .small { font-size: 0.72em; }
.reveal pre, .reveal code { font-size: 0.62em; }
.reveal .flow { text-align: left; font-family: ui-monospace, Consolas, monospace; font-size: 0.62em; line-height: 1.35; background: #f4f7fa; padding: 0.8em 1em; border-left: 4px solid var(--accent); }
.reveal table { font-size: 0.7em; }
.reveal .footer { position: absolute; bottom: 18px; left: 24px; font-size: 0.38em; color: #666; }
.reveal section { text-align: left; }
.reveal ul { margin-left: 1.1em; }
.reveal .center-title { text-align: center; }
.reveal .center-title h1 { font-size: 2.1em; }
"""

REVEAL_HEAD = """<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{title}</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/reveal.js@5.1.0/dist/reset.css"/>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/reveal.js@5.1.0/dist/reveal.css"/>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/reveal.js@5.1.0/dist/theme/white.css"/>
<link rel="stylesheet" href="{css}"/>
</head>
<body>
<div class="reveal"><div class="slides">
"""

REVEAL_TAIL = """
</div></div>
<script src="https://cdn.jsdelivr.net/npm/reveal.js@5.1.0/dist/reveal.js"></script>
<script src="https://cdn.jsdelivr.net/npm/reveal.js@5.1.0/plugin/math/math.js"></script>
<script>
Reveal.initialize({
  hash: true,
  slideNumber: true,
  width: 1280,
  height: 720,
  margin: 0.08,
  plugins: [RevealMath.KaTeX]
});
</script>
</body></html>
"""


def sec(*html_parts: str) -> str:
    return "<section>\n" + "\n".join(html_parts) + "\n<p class='footer'>MC1 · IA científica · DRX · UEM · Dr. Anuar J. Mincache</p>\n</section>"


def deck(title: str, sections: list[str], css_rel: str) -> str:
    return REVEAL_HEAD.format(title=title, css=css_rel) + "\n".join(sections) + REVEAL_TAIL


M1 = [
    sec(
        "<p class='subtitle'>MÓDULO 1</p>",
        "<h1>Inteligência Artificial Científica</h1>",
        "<p>Como usar ML e LLMs <em>junto</em> com a Física — sem substituí-la.</p>",
    ),
    sec(
        "<h2>O que é IA científica</h2>",
        "<ul>",
        "<li>Métodos computacionais acoplados a um problema <strong>físico</strong> bem definido.</li>",
        "<li>Há dados, unidades, hipóteses e <strong>limites de validade</strong>.</li>",
        "<li>Não é “pedir à IA para descobrir a estrutura”.</li>",
        "</ul>",
    ),
    sec(
        "<h2>O que fazemos de fato</h2>",
        "<ol>",
        "<li>Pipeline <strong>reproduzível</strong>.</li>",
        "<li>Grandezas com significado físico (2θ, FWHM, cristalito…).</li>",
        "<li>ML só com pergunta preditiva clara.</li>",
        "<li>LLM para <strong>redigir e explicar</strong> o que já foi calculado.</li>",
        "</ol>",
    ),
    sec(
        "<h2>ML × Deep Learning × LLM</h2>",
        "<table><thead><tr><th>Família</th><th>Papel aqui</th></tr></thead><tbody>",
        "<tr><td>Machine Learning</td><td>Prever % Nd a partir de features do DRX</td></tr>",
        "<tr><td>Deep Learning</td><td>Fora do escopo — só 5 amostras reais</td></tr>",
        "<tr><td>LLM</td><td>Interpretar métricas e escrever o relatório</td></tr>",
        "</tbody></table>",
    ),
    sec(
        "<h2>Pipeline híbrido</h2>",
        "<div class='flow'>DRX → leitura → pré-processamento → features<br>→ ML → avaliação → LLM → relatório</div>",
        "<p>O LLM entra <strong>no fim</strong>, não no lugar do goniômetro.</p>",
    ),
    sec(
        "<h2>Regra de ouro</h2>",
        "<ul>",
        "<li>O LLM <strong>não</strong> calcula 2θ, FWHM, R² nem Scherrer.</li>",
        "<li>Ele recebe um <strong>dossiê numérico</strong> do Python.</li>",
        "<li>Dossiê errado → relatório errado. O cientista responde pelos números.</li>",
        "</ul>",
    ),
]

M2 = [
    sec(
        "<p class='subtitle'>MÓDULO 2</p>",
        "<h1>Dados experimentais de DRX</h1>",
        "<p>Bi<sub>1−x</sub>Nd<sub>x</sub>FeO<sub>3</sub> · x = 10% … 50%</p>",
    ),
    sec(
        "<h2>Bragg (lembrete)</h2>",
        r"<p>\[ n\lambda = 2 d_{hkl} \sin\theta \]</p>",
        "<p class='small'>Pó policristalino · Cu Kα (λ = 1,5406 Å) como hipótese de trabalho neste curso.</p>",
        "<p>O arquivo é o par (2θ, I): a impressão digital cristalina da amostra.</p>",
    ),
    sec(
        "<h2>O que a dopagem pode fazer no DRX</h2>",
        "<ul>",
        "<li><strong>Deslocar</strong> picos → mudança de d<sub>hkl</sub>.</li>",
        "<li><strong>Mudar intensidade</strong> → fator de estrutura, ocupação, textura.</li>",
        "<li><strong>Alargar</strong> picos → cristalito, microdeformação, instrumento.</li>",
        "</ul>",
        "<p class='small'>Isso é expectativa física — ainda não é resultado do notebook.</p>",
    ),
    sec(
        "<h2>Arquivos</h2>",
        "<p><code>Nd_10.csv</code> … <code>Nd_50.csv</code></p>",
        "<ul>",
        "<li>Duas colunas, <strong>sem cabeçalho</strong>: 2θ (°) e I (u.a.).</li>",
        "<li>Varredura típica ~10°–80°.</li>",
        "<li><code>pd.read_csv</code> sem <code>header=None</code> descarta o primeiro ponto. Bug da edição anterior.</li>",
        "</ul>",
    ),
    sec(
        "<h2>Como olhar os dados</h2>",
        "<ol>",
        "<li>Um padrão sozinho (eixos e unidades).</li>",
        "<li>Waterfall das cinco dopagens (comparar formas).</li>",
        "<li>Mapa 2θ × % Nd (picos que caminham).</li>",
        "<li>3D só como complemento — em aula o waterfall ganha.</li>",
        "</ol>",
    ),
    sec(
        "<h2>Leitura qualitativa</h2>",
        "<ul>",
        "<li>Mesmo 2θ → mesma família de planos (hipótese).</li>",
        "<li>Pico que caminha com x → possível variação de rede.</li>",
        "<li>Pico que nasce/some → fase extra ou impureza a checar.</li>",
        "</ul>",
        "<p>Quantificar só depois de limpar ruído e linha de base.</p>",
    ),
]

M3 = [
    sec(
        "<p class='subtitle'>MÓDULO 3</p>",
        "<h1>Pré-processamento</h1>",
        "<p>Do arquivo bruto a um padrão comparável.</p>",
    ),
    sec(
        "<h2>Três etapas</h2>",
        "<ol>",
        "<li><strong>Savitzky–Golay</strong> — suaviza ruído sem deslocar tanto os picos.</li>",
        "<li><strong>ALS</strong> — linha de base assimétrica (picos não são engolidos).</li>",
        "<li><strong>Normalização</strong> — I / I<sub>max</sub> para comparar formas.</li>",
        "</ol>",
    ),
    sec(
        "<h2>O que a normalização apaga</h2>",
        "<p>Contagens absolutas do detector. Bom para forma e posição; ruim se você quiser comparar intensidades absolutas sem calibração.</p>",
        "<p class='small'>Neste curso as cinco medidas são do mesmo tipo de experimento — a normalização é didática para o ML não viciar na escala bruta.</p>",
    ),
    sec(
        "<h2>O que não fazemos aqui</h2>",
        "<ul>",
        "<li>Calibração de 2θ com padrão interno (Si, LaB<sub>6</sub>…).</li>",
        "<li>Correção instrumental do FWHM (Caglioti / padrão).</li>",
        "<li>Refinamento Rietveld.</li>",
        "</ul>",
        "<p>Deslocamentos pequenos podem misturar física e erro de zero do goniômetro.</p>",
    ),
]

M4 = [
    sec(
        "<p class='subtitle'>MÓDULO 4</p>",
        "<h1>Feature engineering</h1>",
        "<p>Milhares de pontos → um vetor curto com leitura física.</p>",
    ),
    sec(
        "<h2>Por que não jogar o espectro inteiro no modelo?</h2>",
        "<p>Isso seria um problema de alta dimensão. Com n = 5, deep learning no padrão bruto só memoriza.</p>",
        "<p>Compactamos: posição, FWHM, área, centroide, n° de picos, estatísticas, Scherrer.</p>",
    ),
    sec(
        "<h2>Características × física</h2>",
        "<table><tbody>",
        "<tr><td>2θ do máximo</td><td>família de planos mais intensa</td></tr>",
        "<tr><td>FWHM</td><td>tamanho + strain + instrumento</td></tr>",
        "<tr><td>Área / centroide</td><td>envelope global</td></tr>",
        "<tr><td>n° de picos</td><td>complexidade / fases</td></tr>",
        "<tr><td>I<sub>max</sub> bruta</td><td>mistura física e detector</td></tr>",
        "</tbody></table>",
    ),
    sec(
        "<h2>Scherrer</h2>",
        r"<p>\[ D = \dfrac{K\lambda}{\beta \cos\theta} \]</p>",
        "<p class='small'>K ≈ 0,9 · λ = 1,5406 Å · β = FWHM em radianos · θ = metade de 2θ.</p>",
        "<p>É ordem de grandeza, <strong>não</strong> Rietveld. Não separa alargamento instrumental nem microdeformação.</p>",
    ),
    sec(
        "<h2>Williamson–Hall (UDM)</h2>",
        r"<p>\[ \beta\cos\theta = K\lambda/D + 4\varepsilon\sin\theta \]</p>",
        "<ul>",
        "<li>Intercepto → tamanho aparente de cristalito D.</li>",
        "<li>Declive → microstrain ε.</li>",
        "<li>β = FWHM de cada pico (não a derivada da intensidade).</li>",
        "<li>Sem padrão interno: D e ε incluem o instrumento.</li>",
        "</ul>",
    ),
    sec(
        "<h2>Catálogo de picos</h2>",
        "<p>Cada pico é um registro: 2θ, I, FWHM, d<sub>hkl</sub>, D(Scherrer).</p>",
        "<p>Cinco CSV viram dezenas de linhas — o máximo de “amostras internas” sem inventar experimento.</p>",
    ),
]

M5 = [
    sec(
        "<p class='subtitle'>MÓDULO 5</p>",
        "<h1>Machine Learning: regressão</h1>",
        "<p>Pergunta: as features do DRX estimam a % de Nd?</p>",
    ),
    sec(
        "<h2>Setup</h2>",
        "<ul>",
        "<li>y = nd_percent (10 … 50)</li>",
        "<li>X = tabela de características (sem o próprio y)</li>",
        "<li>Scaler: features em unidades diferentes (° vs nm vs u.a.)</li>",
        "</ul>",
    ),
    sec(
        "<h2>Por que não 80/20?</h2>",
        "<p>Cinco amostras → o teste teria <strong>um</strong> ponto. Qualquer R² é loteria.</p>",
        "<p><strong>Leave-One-Out:</strong> treina em 4, testa na que ficou de fora, repete. Honesto — e costuma doer.</p>",
    ),
    sec(
        "<h2>Métricas</h2>",
        "<ul>",
        "<li><strong>R²</strong> — no LOO pode ser negativo (pior que a média).</li>",
        "<li><strong>MAE</strong> — erro em pontos percentuais de Nd.</li>",
        "<li><strong>RMSE</strong> — penaliza erros grandes.</li>",
        "</ul>",
        "<p>Compare sempre <em>ajuste in-sample</em> vs LOO. A diferença é o overfitting.</p>",
    ),
    sec(
        "<h2>Modelos na aula</h2>",
        "<ul>",
        "<li>Linear — hiperplano; p ≈ n é perigoso.</li>",
        "<li>k-NN (k = 2) — média dos vizinhos.</li>",
        "<li>SVR (RBF) — superfície suave.</li>",
        "<li>Random Forest — flexível; com n = 5 memoriza.</li>",
        "</ul>",
    ),
    sec(
        "<h2>Dados sintéticos</h2>",
        "<p>Ruído nas 5 medidas <strong>aumenta n</strong> mas <strong>não cria física nova</strong>.</p>",
        "<p>Serve para ensinar o fluxo treino/teste. Não serve para paper.</p>",
    ),
]

M6 = [
    sec(
        "<p class='subtitle'>MÓDULO 6</p>",
        "<h1>LLM como copiloto</h1>",
        "<p>Texto depois dos números — nunca no lugar deles.</p>",
    ),
    sec(
        "<h2>O que o LLM faz</h2>",
        "<ul>",
        "<li>Interpreta o JSON do pipeline.</li>",
        "<li>Explica R², MAE e RMSE <em>neste</em> experimento.</li>",
        "<li>Redige um relatório de minicurso.</li>",
        "</ul>",
    ),
    sec(
        "<h2>O que o LLM não faz</h2>",
        "<ul>",
        "<li>Não mede 2θ.</li>",
        "<li>Não calcula FWHM nem Scherrer.</li>",
        "<li>Não inventa fases que não estejam no dossiê.</li>",
        "</ul>",
        "<p>Se não está no JSON, a resposta honesta é “não consta”.</p>",
    ),
    sec(
        "<h2>Na prática (Colab)</h2>",
        "<ol>",
        "<li>Opcional: Secrets → <code>GEMINI_API_KEY</code>.</li>",
        "<li>Sem chave: relatório local determinístico (a aula não para).</li>",
        "<li>O aluno deve conferir se o texto bate com a tabela.</li>",
        "</ol>",
    ),
]

M7 = [
    sec(
        "<p class='subtitle'>MÓDULO 7</p>",
        "<h1>Pipeline híbrido</h1>",
        "<p>Do CSV ao relatório, numa função só.</p>",
    ),
    sec(
        "<h2>Fluxo completo</h2>",
        "<div class='flow'>Difratograma (DRX)<br>↓ leitura<br>↓ pré-processamento<br>↓ extração de características<br>↓ Machine Learning<br>↓ avaliação (LOO)<br>↓ LLM<br>↓ interpretação<br>↓ relatório final</div>",
    ),
    sec(
        "<h2>Mensagem final</h2>",
        "<p>Se o número não saiu de uma célula de cálculo, ele não entra no artigo — nem quando a frase do LLM estiver convincente.</p>",
        "<p class='small'>Notebook: <code>notebooks/Minicurso_IA_Cientifica_DRX.ipynb</code> · "
        "Slides: <a href='https://raw.githack.com/220719/semana-da-fisica-2025/main/slides/apresentacao.html'>abrir no navegador</a></p>",
    ),
]

INTRO = [
    sec(
        "<div class='center-title'>",
        "<p class='subtitle'>SEMANA DA FÍSICA 2026 · UEM</p>",
        "<h1>IA científica aplicada<br>à difração de raios X</h1>",
        "<p>Dr. Anuar José Mincache<br>Pós-doutorado — Lund University</p>",
        "<p class='small'>Google Colab · dados reais de Bi<sub>1−x</sub>Nd<sub>x</sub>FeO<sub>3</sub></p>",
        "</div>",
    ),
    sec(
        "<h2>Sete módulos</h2>",
        "<ol>",
        "<li>IA científica e pipeline híbrido</li>",
        "<li>Dados experimentais de DRX</li>",
        "<li>Pré-processamento</li>",
        "<li>Feature engineering</li>",
        "<li>Modelos de ML e overfitting</li>",
        "<li>LLM copiloto</li>",
        "<li>Pipeline completo</li>",
        "</ol>",
    ),
]

INDEX = """<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Slides — Minicurso IA científica e DRX</title>
<link rel="stylesheet" href="assets/curso.css"/>
<style>
body { margin: 0; font-family: "Segoe UI", sans-serif; background: #f6f8fb; color: #1a1a1a; }
main { max-width: 720px; margin: 48px auto; padding: 0 24px; }
h1 { color: #0b3d5c; }
a { color: #0b3d5c; }
.card { background: #fff; border: 1px solid #dde3ea; border-radius: 10px; padding: 16px 18px; margin: 10px 0; }
.card a { font-weight: 650; text-decoration: none; }
.card p { margin: 6px 0 0; color: #444; font-size: 0.95em; }
.cta { display: inline-block; margin-top: 8px; background: #0b3d5c; color: #fff !important; padding: 10px 14px; border-radius: 8px; }
</style>
</head>
<body>
<main>
<h1>Slides do minicurso</h1>
<p>Abra no navegador. Setas do teclado avançam os slides. F = tela cheia (Reveal.js).</p>
<p><a class="cta" href="apresentacao.html">Abrir apresentação completa</a></p>
<div class="card"><a href="01-ia-cientifica.html">Módulo 1 — IA científica</a><p>ML, DL, LLM e a regra de ouro do pipeline híbrido.</p></div>
<div class="card"><a href="02-dados-experimentais.html">Módulo 2 — Dados experimentais</a><p>Bragg, arquivos CSV, waterfall e leitura qualitativa.</p></div>
<div class="card"><a href="03-preprocessamento.html">Módulo 3 — Pré-processamento</a><p>Savitzky–Golay, ALS e normalização.</p></div>
<div class="card"><a href="04-features.html">Módulo 4 — Features</a><p>Picos, FWHM, Scherrer e o que tem significado físico.</p></div>
<div class="card"><a href="05-machine-learning.html">Módulo 5 — Machine Learning</a><p>LOO, métricas, overfitting e dados sintéticos.</p></div>
<div class="card"><a href="06-llm-copiloto.html">Módulo 6 — LLM copiloto</a><p>O modelo de linguagem só interpreta o JSON.</p></div>
<div class="card"><a href="07-pipeline-hibrido.html">Módulo 7 — Pipeline híbrido</a><p>Do difratograma ao relatório final.</p></div>
</main>
</body>
</html>
"""


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    ASSETS.mkdir(parents=True, exist_ok=True)
    (ASSETS / "curso.css").write_text(CSS, encoding="utf-8")
    (ROOT / "index.html").write_text(INDEX, encoding="utf-8")

    parts = [
        ("01-ia-cientifica.html", "Módulo 1 — IA científica", M1, "assets/curso.css"),
        ("02-dados-experimentais.html", "Módulo 2 — Dados experimentais", M2, "assets/curso.css"),
        ("03-preprocessamento.html", "Módulo 3 — Pré-processamento", M3, "assets/curso.css"),
        ("04-features.html", "Módulo 4 — Features", M4, "assets/curso.css"),
        ("05-machine-learning.html", "Módulo 5 — Machine Learning", M5, "assets/curso.css"),
        ("06-llm-copiloto.html", "Módulo 6 — LLM copiloto", M6, "assets/curso.css"),
        ("07-pipeline-hibrido.html", "Módulo 7 — Pipeline híbrido", M7, "assets/curso.css"),
    ]
    for name, title, sections, css in parts:
        (ROOT / name).write_text(deck(title, sections, css), encoding="utf-8")

    all_sec = INTRO + M1 + M2 + M3 + M4 + M5 + M6 + M7
    (ROOT / "apresentacao.html").write_text(
        deck("Minicurso IA científica e DRX", all_sec, "assets/curso.css"),
        encoding="utf-8",
    )
    print(f"Wrote slides in {ROOT}")


if __name__ == "__main__":
    main()
