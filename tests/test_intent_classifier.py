import os
import sys
import pytest
import numpy as np
import yaml
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from dotenv import load_dotenv

# Adiciona o diretório raiz ao sys.path para encontrar o módulo
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Importa o código fonte que estamos testando
from intent_classifier import IntentClassifier, Config

# --- Fixtures (Contextos de Teste) ---

@pytest.fixture(scope="session")
def paths():
    """Fornece caminhos para arquivos de dados de teste, pulando se não encontrados."""
    test_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(test_dir, ".."))
    
    # Caminhos baseados na estrutura de arquivos original
    examples_path = os.path.join(project_root, "intent_classifier", "data", "confusion_intents.yml")
    config_path = os.path.join(project_root, "intent_classifier", "models", "confusion-v1_config.yml")

    if not (os.path.exists(examples_path)):
        pytest.skip(f"Arquivos de dados de teste não encontrados em {examples_path}")
    if not (os.path.exists(config_path)):
        pytest.skip(f"Arquivos de configuração não encontrados em {config_path}")
        
    return {"config": config_path, "examples": examples_path}

@pytest.fixture(scope="module")
def clf_wandb(paths):
    """
    Fixture de integração: Carrega o modelo real do W&B.
    - Pula (skip) se WANDB_MODEL_URL ou WANDB_API_KEY não estiverem definidos.
    - Falha (fail) se as secrets estiverem definidas, mas o modelo não carregar.
    """
    load_dotenv()
    env_url = os.getenv("WANDB_MODEL_URL")
    api_key = os.getenv("WANDB_API_KEY")

    if not api_key:
        pytest.skip("WANDB_API_KEY não definido. Pulando teste de integração W&B.")
    if not env_url:
        pytest.skip("WANDB_MODEL_URL não definido. Pulando teste de integração W&B.")

    print("\n🌐 WANDB_MODEL_URL detectado, tentando carregar modelo real...")

    # A inicialização tentará carregar o modelo
    classifier = IntentClassifier(config=paths["config"], 
                                  examples_file=paths["examples"],
                                  load_model=os.environ.get('WANDB_MODEL_URL'))
    
    # Validação crucial: o __init__ do código-fonte captura exceções e apenas imprime.
    # Devemos verificar ativamente se o modelo foi carregado.
    if classifier.model is None:
        pytest.fail(
            f"Secrets W&B definidas, mas o modelo falhou ao carregar de {env_url}. "
            "Verifique a API key, a URL do modelo ou se o modelo existe."
        )
        
    print("✅ Modelo carregado do W&B")
    return classifier

@pytest.fixture(scope="module")
def clf_local_trained(paths):
    """
    Fixture de sanidade: Treina um modelo pequeno localmente.
    Usa a sugestão de treinar com dados locais para um teste e2e.
    """
    print("\n⚙️ Treinando modelo local para testes...")
    
    # Configuração mínima para um treino rápido
    local_config = Config(
        dataset_name="local_test",
        epochs=2,  # Apenas 2 épocas para velocidade
        callback_patience=1,
        validation_split=0.5,
        sent_hl_units=8,  # Modelo pequeno
        wandb_project=None  # Não registrar este treino de teste
    )
    
    # Passa o objeto Config e o caminho dos exemplos
    classifier = IntentClassifier(config=local_config, examples_file=paths["examples"])
    classifier.train(tf_verbosity=0)  # Treina silenciosamente
    print("✅ Modelo local treinado.")
    return classifier

@pytest.fixture
def clf_minimal():
    """Classificador leve, sem modelo, para testes de unidade rápidos."""
    # Fornece 'codes' para satisfazer o _setup_encoder
    config = Config(
        dataset_name="minimal_test",
        min_words=2,
        codes=["intent_a", "intent_b"]
    )
    return IntentClassifier(config=config)

@pytest.fixture
def clf_with_stopwords(tmp_path):
    """Classificador leve com um arquivo de stopwords temporário."""
    # tmp_path é uma fixture nativa do pytest
    stop_words_file = tmp_path / "stopwords.txt"
    stop_words_file.write_text("um\numa\nde\ndo")
    
    config = Config(
        dataset_name="stopwords_test",
        codes=["intent_a"],
        stop_words_file=str(stop_words_file)
    )
    return IntentClassifier(config=config)

# --- Testes de Unidade (Rápidos) ---

def test_init_fails_without_config_or_model(monkeypatch):
    """Verifica se a inicialização falha se NENHUMA fonte de config/modelo for fornecida."""
    # monkeypatch é uma fixture nativa do pytest para alterar o ambiente
    monkeypatch.delenv("WANDB_MODEL_URL", raising=False)
    
    # Deve falhar, pois nem 'config', nem 'load_model', nem WANDB_MODEL_URL foram dados
    with pytest.raises(ValueError, match="`config` object must be provided"):
        IntentClassifier()

def test_preprocess_text_lowercase(clf_minimal):
    """Testa a conversão para minúsculas."""
    result_tensor = clf_minimal.preprocess_text("OI TUDO BEM?")
    assert result_tensor.shape == (1,)
    assert result_tensor.numpy()[0] == b'oi tudo bem?'

def test_preprocess_text_min_words(clf_minimal):
    """Testa o padding de 'min_words'."""
    # A config em 'clf_minimal' define min_words=2
    # O código fonte adiciona (min_words + 1) de padding
    result_tensor = clf_minimal.preprocess_text("oi") # Menor que min_words
    assert result_tensor.numpy()[0] == b'<> <> <>'

def test_preprocess_text_stopwords(clf_with_stopwords):
    """Testa a remoção de stopwords."""
    result_tensor = clf_with_stopwords.preprocess_text("uma frase de teste")
    assert result_tensor.numpy()[0] == b'frase teste'

# --- Testes de Sanidade Local (Médios) ---

def test_local_train_model_created(clf_local_trained):
    """Verifica se o modelo local foi treinado e atribuído."""
    assert clf_local_trained.model is not None
    assert isinstance(clf_local_trained.model, tf.keras.Model)

def test_local_predict_sanity(clf_local_trained):
    """Testa a previsão (predict) usando o modelo treinado localmente."""
    clf = clf_local_trained
    top_intent, probs = clf.predict("oi como vai")
    
    assert isinstance(top_intent, str)
    assert top_intent in clf.codes
    assert isinstance(probs, dict)
    assert list(probs.keys()) == list(clf.codes)
    assert sum(probs.values()) == pytest.approx(1.0)

def test_one_hot_encoder_local(clf_local_trained):
    """Valida o one-hot encoder usando o modelo local treinado."""
    clf = clf_local_trained
    enc = clf.onehot_encoder
    codes = list(clf.codes)
    
    assert len(codes) > 1  # Garante que os dados de teste foram carregados
    
    for idx, code in enumerate(codes):
        vec = enc.transform([[code]]).toarray()[0]
        assert len(vec) == len(codes)
        assert vec[idx] == pytest.approx(1.0)
        assert ((vec == 0) | (vec == 1)).all()
        decoded = enc.inverse_transform([vec])[0][0]
        assert decoded == code

# --- Testes de Integração (Lentos) --- podem ser pulados se executar: `pytest -m "not integration"`

@pytest.mark.integration
def test_wandb_model_predicts(clf_wandb):
    """
    Teste de integração real:
    Verifica se o modelo carregado do W&B consegue fazer uma previsão.
    """
    print("🔎 Verificando a previsão (predict) do modelo do W&B")
    top_intent, probs = clf_wandb.predict("exemplo qualquer")
    
    print(f"Intenção prevista: {top_intent}")
    assert isinstance(top_intent, str)
    assert isinstance(probs, dict)
    assert len(probs) >= 1
    assert top_intent in clf_wandb.codes

@pytest.mark.integration
def test_wandb_model_accuracy_easy_examples(clf_wandb, paths):
    """
    Teste de integração real:
    Verifica a acurácia do modelo do W&B em exemplos conhecidos.
    """
    print("🌐 Usando modelo do W&B para verificação de acurácia")
    
    with open(paths["examples"], "r") as f:
        data = yaml.safe_load(f)

    print(f"📂 Carregando exemplos de {paths['examples']}")
    samples = []
    for intent_block in data:
        for text in intent_block["examples"]:
            samples.append((text, intent_block["intent"]))
    
    # Pega uma amostra (máx 50) para não demorar muito
    samples = samples[:50] 
    texts = [t for t, _ in samples]
    labels = [l for _, l in samples]
    
    preds = clf_wandb.predict(texts)
    pred_labels = [p[0] for p in preds]

    accuracy = sum(p == l for p, l in zip(pred_labels, labels)) / len(labels)
    print(f"🏆 Acurácia na amostra: {accuracy:.2f}")

    # Relatório e Matriz de Confusão
    report = classification_report(labels, pred_labels, zero_division=0)
    print("\n📄 Relatório de Classificação (Amostra):\n" + report)

    all_labels = sorted(set(labels) | set(pred_labels))
    cm = confusion_matrix(labels, pred_labels, labels=all_labels)
    cm_df = pd.DataFrame(cm, index=all_labels, columns=all_labels)
    print("\n🔢 Matriz de Confusão (Amostra):\n" + cm_df.to_string())

    # Define um limite de acurácia razoável para o modelo de produção
    assert accuracy >= 0.7