import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import os
import json
from loguru import logger
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.impute import KNNImputer
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import shap
import base64
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import subprocess
import requests
from transformers import pipeline, BioGptTokenizer, BioGptForCausalLM
from transformers import MarianMTModel, MarianTokenizer

# Configure logging
logger.add("logs/omics_predictor.log", rotation="10 MB", level="INFO",
           format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")


# Load translations
def load_translations():
    try:
        with open("translations.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error("translations.json file not found")
        st.error("Translation file not found.")
        return {}


translations = load_translations()


# Get translation function
def t(key, lang="fr"):
    return translations.get(lang, {}).get(key, key)


# Set page configuration
st.set_page_config(page_title=t("app_title"), layout="wide", initial_sidebar_state="expanded")


# Load custom CSS
def load_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.error(t("css_error"))
        logger.error("style.css file not found")


load_css("style.css")

# Initialize session state
if 'results' not in st.session_state:
    st.session_state.results = None
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = {}
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'patient_id' not in st.session_state:
    st.session_state.patient_id = None
if 'language' not in st.session_state:
    st.session_state.language = "fr"

# Database setup (SQLite)
Base = declarative_base()


class Patient(Base):
    __tablename__ = 'patients'
    id = Column(Integer, primary_key=True)
    patient_id = Column(String, unique=True)
    risk_score = Column(Float)
    biomarkers = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)


engine = create_engine('sqlite:///data/patients.db')
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)

# Sidebar
with st.sidebar:
    st.markdown(f"<div class='sidebar-header'>{t('app_title')}</div>", unsafe_allow_html=True)
    st.image("https://via.placeholder.com/150", caption=t("author"))
    st.markdown(f"<div class='sidebar-subheader'>{t('navigation')}</div>", unsafe_allow_html=True)

    # Language selector
    lang = st.selectbox(t("select_language"), ["Français", "English"], key="lang_select")
    st.session_state.language = "fr" if lang == "Français" else "en"

    page = st.selectbox(t("choose_page"), [
        t("home"), t("about_omics"), t("predict_risk"), t("chatbot"), t("dashboard"), t("logs")
    ], key="nav")
    st.markdown(f"<div class='sidebar-footer'>{t('footer')}</div>", unsafe_allow_html=True)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

# IRC biomarkers
irc_biomarkers = [
    'UMOD_rs12917707', 'APOL1_rs73885319', 'MYH9_rs4821480', 'HAVCR1', 'TGFB1', 'IL6', 'HNF4A', 'NPHS1', 'AQP2',
    'B2MG', 'Albumin', 'NGAL', 'Cystatin_C', 'Uromodulin', 'KLOTHO', 'Kynurenine', 'Indoxyl_Sulfate', 'Creatinine',
    '5-MTP'
]


# Model definition
class OmicsVAE(nn.Module):
    def __init__(self, input_dims, hidden_dim=256, latent_dim=64, num_heads=8, num_layers=3, dropout=0.4,
                 num_classes=2):
        super(OmicsVAE, self).__init__()
        self.input_dims = input_dims
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_omics = len(input_dims)
        self.num_classes = num_classes

        self.input_projections = nn.ModuleList([nn.Linear(dim, hidden_dim) for dim in input_dims])
        self.positional_encoding = self.create_positional_encoding(hidden_dim, max_len=self.num_omics)
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 4, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=num_layers)

        self.fc_mu = nn.Linear(hidden_dim * self.num_omics, latent_dim)
        self.fc_log_var = nn.Linear(hidden_dim * self.num_omics, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, hidden_dim * self.num_omics)
        self.decoder_projections = nn.ModuleList([nn.Linear(hidden_dim, dim) for dim in input_dims])
        self.fc_classify = nn.Linear(latent_dim, num_classes)

    def create_positional_encoding(self, d_model, max_len):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x_list):
        encoded = []
        for i, x in enumerate(x_list):
            proj = self.input_projections[i](x)
            pe = self.positional_encoding[:, i, :].to(x.device)
            proj = proj + pe.expand(x.size(0), -1)
            encoded.append(proj.unsqueeze(1))
        encoded = torch.cat(encoded, dim=1)
        transformer_out = self.transformer_encoder(encoded)
        transformer_out = transformer_out.contiguous().view(transformer_out.size(0), -1)

        mu = self.fc_mu(transformer_out)
        log_var = self.fc_log_var(transformer_out)
        z = self.reparameterize(mu, log_var)

        decoded = self.fc_decode(z).view(z.size(0), self.num_omics, self.hidden_dim)
        outputs = [self.decoder_projections[i](decoded[:, i, :]) for i in range(self.num_omics)]
        class_logits = self.fc_classify(z)
        return outputs, z, mu, log_var, class_logits


# Data loading and preprocessing
@st.cache_data
def load_omics_data(file_paths):
    try:
        data_dict = {}
        for omic, path in file_paths.items():
            logger.info(f"Loading {omic} file: {path}")
            df = pd.read_csv(path, index_col='Patient_ID')
            if 'Status' in df.columns:
                df = df.drop(columns=['Status'])
            data_dict[omic] = df
        labels = pd.Series(['Unknown'] * len(data_dict['genomics']), index=data_dict['genomics'].index, name='Status')
        le = LabelEncoder()
        le.fit(['Unknown', 'IRC', 'Non-IRC'])
        encoded_labels = pd.Series(le.transform(labels), index=labels.index, name='Status')
        return data_dict, encoded_labels, le
    except Exception as e:
        logger.error(f"Data loading error: {str(e)}")
        raise Exception(t("data_load_error").format(error=str(e)))


@st.cache_data
def preprocess_omics(data_dict, genotype_mapping={'AA': 0, 'AG': 1, 'GG': 2}):
    imputer = KNNImputer(n_neighbors=5)
    scaler = RobustScaler()
    for omic in data_dict:
        logger.info(f"Preprocessing {omic} data")
        try:
            if omic == 'genomics':
                data_dict[omic] = data_dict[omic].apply(lambda x: x.map(genotype_mapping))
            if not data_dict[omic].select_dtypes(include=[np.number]).shape[1] == data_dict[omic].shape[1]:
                raise ValueError(f"Non-numeric data in {omic}")
            data_imputed = pd.DataFrame(
                imputer.fit_transform(data_dict[omic]),
                columns=data_dict[omic].columns,
                index=data_dict[omic].index
            )
            data_scaled = pd.DataFrame(
                scaler.fit_transform(data_imputed),
                columns=data_imputed.columns,
                index=data_imputed.index
            )
            data_dict[omic] = data_scaled
        except Exception as e:
            logger.error(f"Preprocessing error for {omic}: {str(e)}")
            raise Exception(t("preprocess_error").format(omic=omic, error=str(e)))
    return data_dict


# Latent representation extraction
def extract_latent_representations(model, dataloader, device):
    encoded_data = []
    try:
        with torch.no_grad():
            for batch in dataloader:
                inputs = [b.to(device) for b in batch]
                _, z, _, _, _ = model(inputs)
                encoded_data.append(z.cpu().numpy())
        return np.concatenate(encoded_data, axis=0)
    except Exception as e:
        logger.error(f"Latent extraction error: {str(e)}")
        raise Exception(t("latent_error").format(error=str(e)))


# Risk score calculation
def calculate_risk_scores(encoded_data, data_dict, irc_biomarkers, labels, label_encoder):
    try:
        umap_df = pd.DataFrame(index=data_dict['genomics'].index)
        base_risk = np.zeros(len(encoded_data))

        biomarker_weights = {omic: {} for omic in data_dict}
        for omic in data_dict:
            for col in data_dict[omic].columns:
                weight = 2.0 if col in irc_biomarkers else 1.0
                biomarker_weights[omic][col] = weight

        weighted_risk = base_risk.copy()
        for omic in data_dict:
            for col, weight in biomarker_weights[omic].items():
                deviation = np.abs(data_dict[omic][col].values - data_dict[omic][col].mean())
                weighted_risk += weight * deviation

        weighted_risk = (weighted_risk - weighted_risk.min()) / (weighted_risk.max() - weighted_risk.min())

        classifier = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        scaler = StandardScaler()
        encoded_scaled = scaler.fit_transform(encoded_data)
        classifier.fit(encoded_scaled, labels)
        irc_class = label_encoder.transform(['IRC'])[0]
        class_probs = classifier.predict_proba(encoded_scaled)[:, irc_class]
        final_risk = weighted_risk * 0.5 + class_probs * 0.5
        final_risk = final_risk * 100
        umap_df[t('risk_score')] = final_risk
        logger.info("Risk scores calculated successfully")
        return umap_df
    except Exception as e:
        logger.error(f"Risk score calculation error: {str(e)}")
        raise Exception(t("risk_score_error").format(error=str(e)))


# SHAP analysis
def perform_shap_analysis(model, combined_data, input_dims, device, feature_names, output_dir, irc_biomarkers):
    try:
        logger.info("Starting SHAP analysis")
        X_concat = combined_data.values
        n_samples = min(100, X_concat.shape[0])
        X_subset = X_concat[:n_samples]

        class VAEWrapper:
            def __init__(self, model, device):
                self.model = model
                self.device = device

            def predict(self, X):
                X_tensors = []
                start = 0
                for dim in input_dims:
                    X_tensors.append(torch.tensor(X[:, start:start + dim], dtype=torch.float32).to(self.device))
                    start += dim
                with torch.no_grad():
                    _, z, _, _, _ = self.model(X_tensors)
                z_norm = torch.norm(z, dim=1).cpu().numpy()
                return z_norm

        explainer = shap.KernelExplainer(VAEWrapper(model, device).predict, X_subset)
        shap_values = explainer.shap_values(X_subset, nsamples=200)

        if not isinstance(shap_values, list):
            shap_values = [shap_values]

        shap_mean = np.mean(np.abs(shap_values[0]), axis=0)
        if shap_mean.ndim > 1:
            shap_mean = shap_mean.flatten()

        shap_importance = pd.DataFrame({
            t('feature'): feature_names[:len(shap_mean)],
            'SHAP_Mean_Abs': shap_mean
        }).sort_values('SHAP_Mean_Abs', ascending=False)

        shap_importance[t('omic')] = [
            t('genomics') if 'rs' in f else t('transcriptomics') if f in ['HAVCR1', 'TGFB1', 'IL6', 'HNF4A', 'NPHS1',
                                                                          'AQP2']
            else t('proteomics') if f in ['B2MG', 'Albumin', 'NGAL', 'Cystatin_C', 'Uromodulin', 'KLOTHO']
            else t('metabolomics') for f in shap_importance[t('feature')]
        ]

        plt.figure(figsize=(12, 8))
        sns.barplot(data=shap_importance.head(20), x='SHAP_Mean_Abs', y=t('feature'), hue=t('omic'), palette='husl')
        plt.title(t('shap_top_features'), fontsize=16)
        plt.xlabel(t('shap_mean_value'), fontsize=12)
        plt.ylabel(t('feature'), fontsize=12)
        plt.legend(title=t('omics'))
        shap_plot_path = os.path.join(output_dir, 'shap_importance_by_omics.png')
        plt.savefig(shap_plot_path, dpi=300)
        plt.close()

        important_biomarkers = shap_importance[shap_importance[t('feature')].isin(irc_biomarkers)]
        important_biomarkers.to_csv(os.path.join(output_dir, 'important_biomarkers.csv'))
        return important_biomarkers, shap_plot_path
    except Exception as e:
        logger.error(f"SHAP analysis error: {str(e)}")
        raise Exception(t("shap_error").format(error=str(e)))


# Generate PDF report
def generate_pdf_report(risk_score, important_biomarkers, shap_plot_path, recommendations, output_path, lang):
    try:
        c = canvas.Canvas(output_path, pagesize=letter)
        c.setFont("Helvetica-Bold", 16)
        c.drawString(100, 750, t('report_title', lang))
        c.setFont("Helvetica", 12)
        c.drawString(100, 730, t('author', lang))
        c.drawString(100, 710, t('footer', lang))

        c.setFont("Helvetica-Bold", 14)
        c.drawString(100, 680, t('ckd_risk_score', lang).format(score=risk_score))
        c.setFont("Helvetica", 12)
        c.drawString(100, 660, t('key_biomarkers', lang))

        y = 640
        for idx, row in important_biomarkers.head(10).iterrows():
            c.drawString(120, y, f"{row[t('feature')]} ({row[t('omic')]}): {row['SHAP_Mean_Abs']:.4f}")
            y -= 20
            if y < 100:
                c.showPage()
                y = 750

        c.setFont("Helvetica-Bold", 14)
        c.drawString(100, y - 20, t('recommendations', lang))
        y -= 40
        for rec in recommendations[:5]:
            c.drawString(120, y, f"- {rec}")
            y -= 20
            if y < 100:
                c.showPage()
                y = 750

        if os.path.exists(shap_plot_path):
            c.drawImage(shap_plot_path, 100, y - 300, width=400, height=200)
        c.save()
        logger.info(f"PDF report generated: {output_path}")
    except Exception as e:
        logger.error(f"PDF report generation error: {str(e)}")
        raise Exception(t("pdf_error").format(error=str(e)))


# Load BioGPT for recommendations
try:
    biogpt_tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")
    biogpt_model = BioGptForCausalLM.from_pretrained("microsoft/biogpt").to(device)
    biogpt_model.eval()
    logger.info("BioGPT model loaded successfully")
except Exception as e:
    st.error(t("biogpt_load_error"))
    logger.error(f"BioGPT load error: {str(e)}")
    st.stop()

# Load translation model (English to French)
try:
    translator_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-fr")
    translator_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-fr").to(device)
    logger.info("Translation model loaded successfully")
except Exception as e:
    st.error(t("translator_load_error"))
    logger.error(f"Translation model load error: {str(e)}")
    st.stop()


# Generate recommendation using BioGPT
@st.cache_data
def generate_recommendation(prompt, max_length=100):
    try:
        inputs = biogpt_tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(device)
        outputs = biogpt_model.generate(
            inputs["input_ids"],
            max_length=max_length,
            num_return_sequences=1,
            do_sample=True,
            top_p=0.9,
            temperature=0.7
        )
        recommendation = biogpt_tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"Generated recommendation: {recommendation}")
        return recommendation
    except Exception as e:
        logger.error(f"BioGPT generation error: {str(e)}")
        return t("recommendation_error")


# Translate recommendation if needed
def translate_recommendation(text, lang):
    if lang == "fr":
        try:
            inputs = translator_tokenizer(text, return_tensors="pt", padding=True).to(device)
            translated = translator_model.generate(**inputs)
            translated_text = translator_tokenizer.decode(translated[0], skip_special_tokens=True)
            logger.info(f"Translated recommendation to French: {translated_text}")
            return translated_text
        except Exception as e:
            logger.error(f"Translation error: {str(e)}")
            return text
    return text


# Validate recommendation with BioBERT
bio_bert = pipeline("text-classification", model="monologg/biobert_v1.1_pubmed", device=-1)


def validate_recommendation(text):
    try:
        result = bio_bert(text)
        is_valid = result[0]['label'] == 'POSITIVE' and result[0]['score'] > 0.9
        logger.info(f"Recommendation validation: {text} -> Valid={is_valid}")
        return is_valid
    except Exception as e:
        logger.error(f"Recommendation validation error: {str(e)}")
        return False


# Formulate prompt for BioGPT
def formulate_prompt(intent, risk_score, biomarkers):
    prompt = f"Patient with Chronic Kidney Disease (CKD) risk score: {risk_score:.2f}%. Key biomarkers: {', '.join(biomarkers)}. "
    if intent == "medication":
        prompt += "Suggest a safe and effective medication treatment for CKD, including dosage and precautions."
    elif intent == "diet":
        prompt += "Recommend a dietary plan for CKD management, including specific foods to include or avoid."
    elif intent == "exercise":
        prompt += "Propose a physical activity plan for CKD, specifying type, duration, and intensity."
    return prompt


# Rasa chatbot integration
def start_rasa_server():
    try:
        process = subprocess.Popen(["rasa", "run", "--enable-api", "--cors", "*"], stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE)
        logger.info("Rasa server started")
        return process
    except Exception as e:
        logger.error(f"Rasa server start error: {str(e)}")
        st.error(t("rasa_error"))
        return None


def send_to_rasa(message, lang):
    try:
        url = "http://localhost:5005/webhooks/rest/webhook"
        payload = {"sender": "user", "message": message, "metadata": {"language": lang}}
        response = requests.post(url, json=payload)
        response_json = response.json()
        if response_json:
            return response_json[0].get('text', t('chatbot_no_response'))
        return t('chatbot_no_response')
    except Exception as e:
        logger.error(f"Rasa communication error: {str(e)}")
        return t('chatbot_comm_error')


# Load model
input_dims = [100, 50, 30, 20]  # Adjust based on your data
best_hyperparams = {
    'hidden_dim': 256, 'latent_dim': 64, 'num_layers': 3, 'dropout': 0.4, 'batch_size': 32
}
try:
    model = OmicsVAE(input_dims=input_dims, **best_hyperparams).to(device)
    model.load_state_dict(torch.load("omics_vae_best_hyperparams.pth", map_location=device))
    model.eval()
    logger.info("Model loaded successfully")
except FileNotFoundError:
    st.error(t("model_weights_error"))
    logger.error("Model weights not found")
    st.stop()
except Exception as e:
    st.error(t("model_load_error").format(error=str(e)))
    logger.error(f"Model load error: {str(e)}")
    st.stop()

# Start Rasa server
rasa_process = start_rasa_server()

# Current language
lang = st.session_state.language

# Home Page
if page == t("home"):
    st.markdown(f"<div class='title animate__fadeIn'>{t('app_title')}</div>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class='intro animate__fadeIn'>
        <p>{t('home_intro_1').format(author="<b>Ngoue David Roger Yannick</b>")}</p>
        <p>{t('home_intro_2')}</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"<div class='section-header'>{t('why_omics')}</div>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class='content animate__fadeIn'>
        <p>{t('omics_description')}</p>
        <ul>
            <li><b>{t('genomics')}</b>: {t('genomics_desc')}</li>
            <li><b>{t('transcriptomics')}</b>: {t('transcriptomics_desc')}</li>
            <li><b>{t('proteomics')}</b>: {t('proteomics_desc')}</li>
            <li><b>{t('metabolomics')}</b>: {t('metabolomics_desc')}</li>
        </ul>
        <p>{t('omics_integration')}</p>
    </div>
    """, unsafe_allow_html=True)

# About Omics Data Page
elif page == t("about_omics"):
    st.markdown(f"<div class='title animate__fadeIn'>{t('about_omics_title')}</div>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class='content animate__fadeIn'>
        <p>{t('about_omics_intro')}</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"<div class='section-header'>{t('raw_omics_formats')}</div>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class='content animate__fadeIn'>
        <p>{t('raw_omics_desc')}</p>
        <ul>
            <li><b>{t('genomics')}</b>: {t('genomics_format')}</li>
            <li><b>{t('transcriptomics')}</b>: {t('transcriptomics_format')}</li>
            <li><b>{t('proteomics')}</b>: {t('proteomics_format')}</li>
            <li><b>{t('metabolomics')}</b>: {t('metabolomics_format')}</li>
        </ul>
        <p>{t('raw_omics_processing')}</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"<div class='section-header'>{t('preprocessing_steps')}</div>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class='content animate__fadeIn'>
        <p>{t('preprocessing_desc')}</p>
        <ol>
            <li><b>{t('quality_control')}</b>: {t('quality_control_desc')}</li>
            <li><b>{t('normalization')}</b>: {t('normalization_desc')}</li>
            <li><b>{t('imputation')}</b>: {t('imputation_desc')}</li>
            <li><b>{t('standardization')}</b>: {t('standardization_desc')}</li>
            <li><b>{t('feature_selection')}</b>: {t('feature_selection_desc')}</li>
        </ol>
        <p>{t('preprocessing_importance')}</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"<div class='section-header'>{t('csv_format')}</div>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class='content animate__fadeIn'>
        <p>{t('csv_format_desc')}</p>
        <ul>
            <li><b>{t('columns')}</b>: {t('columns_desc')}</li>
            <li><b>{t('rows')}</b>: {t('rows_desc')}</li>
            <li><b>{t('values')}</b>: {t('values_desc')}</li>
            <li><b>{t('separator')}</b>: {t('separator_desc')}</li>
            <li><b>{t('header')}</b>: {t('header_desc')}</li>
        </ul>
        <p>{t('csv_example')}</p>
        <pre class='code-block'>
Patient_ID,UMOD_rs12917707,APOL1_rs73885319,MYH9_rs4821480,Status
P001,0,1,2,Unknown
        </pre>
        <p>{t('csv_patient_id')}</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"<div class='section-header'>{t('data_quality')}</div>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class='content animate__fadeIn'>
        <p>{t('data_quality_desc')}</p>
        <ul>
            <li><b>{t('consistency')}</b>: {t('consistency_desc')}</li>
            <li><b>{t('completeness')}</b>: {t('completeness_desc')}</li>
            <li><b>{t('accuracy')}</b>: {t('accuracy_desc')}</li>
            <li><b>{t('alignment')}</b>: {t('alignment_desc')}</li>
        </ul>
        <p>{t('data_quality_importance')}</p>
    </div>
    """, unsafe_allow_html=True)

# Predict Risk Page
elif page == t("predict_risk"):
    st.markdown(f"<div class='title animate__fadeIn'>{t('predict_risk_title')}</div>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class='content animate__fadeIn'>
        <p>{t('predict_risk_intro')}</p>
    </div>
    """, unsafe_allow_html=True)

    # Patient ID input
    patient_id = st.text_input(t("patient_id"), key="patient_id_input")
    if patient_id:
        st.session_state.patient_id = patient_id

    # File upload section
    st.markdown(f"<div class='section-header'>{t('upload_omics')}</div>", unsafe_allow_html=True)
    file_paths = {
        'genomics': None,
        'transcriptomics': None,
        'proteomics': None,
        'metabolomics': None
    }
    with st.container():
        cols = st.columns(4)
        for idx, omic in enumerate(file_paths):
            with cols[idx]:
                uploaded_file = st.file_uploader(f"{t(omic)} (CSV)", type="csv", key=omic)
                if uploaded_file:
                    temp_path = f"temp_{omic}.csv"
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    file_paths[omic] = temp_path
                    st.session_state.uploaded_files[omic] = temp_path

    # Validate and process files
    if st.button(t("run_prediction"), key="run_prediction"):
        if all(file_paths[omic] for omic in file_paths) and st.session_state.patient_id:
            try:
                with st.spinner(t("processing")):
                    # Load and preprocess data
                    data_dict, labels, label_encoder = load_omics_data(file_paths)
                    data_dict = preprocess_omics(data_dict)
                    combined_data = pd.concat([data_dict[omic] for omic in data_dict], axis=1)
                    feature_names = sum([data_dict[omic].columns.tolist() for omic in data_dict], [])

                    # Prepare DataLoader
                    patient_tensors = [torch.tensor(data_dict[omic].values, dtype=torch.float32) for omic in data_dict]
                    patient_dataset = TensorDataset(*patient_tensors)
                    patient_loader = DataLoader(patient_dataset, batch_size=best_hyperparams['batch_size'],
                                                shuffle=False)

                    # Extract latent representations
                    encoded_data = extract_latent_representations(model, patient_loader, device)

                    # Calculate risk scores
                    umap_df = calculate_risk_scores(encoded_data, data_dict, irc_biomarkers, labels, label_encoder)

                    # Perform SHAP analysis
                    output_dir = "temp_output"
                    os.makedirs(output_dir, exist_ok=True)
                    important_biomarkers, shap_plot_path = perform_shap_analysis(
                        model, combined_data, input_dims, device, feature_names, output_dir, irc_biomarkers
                    )

                    # Generate dynamic recommendations
                    risk_score = umap_df[t('risk_score')].iloc[0]
                    biomarkers = important_biomarkers[t('feature')].head(5).tolist()
                    recommendations = []
                    for intent in ["medication", "diet", "exercise"]:
                        prompt = formulate_prompt(intent, risk_score, biomarkers)
                        rec = generate_recommendation(prompt)
                        if validate_recommendation(rec):
                            rec = translate_recommendation(rec, lang)
                            recommendations.append(rec)
                        else:
                            recommendations.append(t(f"{intent}_error", lang))

                    # Generate PDF report
                    pdf_path = os.path.join(output_dir, f"ckd_risk_report_{st.session_state.patient_id}.pdf")
                    generate_pdf_report(risk_score, important_biomarkers, shap_plot_path, recommendations, pdf_path,
                                        lang)

                    # Save to database
                    session = Session()
                    patient = Patient(
                        patient_id=st.session_state.patient_id,
                        risk_score=risk_score,
                        biomarkers=', '.join(biomarkers)
                    )
                    session.add(patient)
                    session.commit()
                    session.close()
                    logger.info(f"Patient {st.session_state.patient_id} saved to database")

                    # Store results
                    st.session_state.results = {
                        'umap_df': umap_df,
                        'important_biomarkers': important_biomarkers,
                        'output_dir': output_dir,
                        'pdf_path': pdf_path,
                        'recommendations': recommendations
                    }

                # Display results
                st.markdown(f"<div class='section-header'>{t('prediction_results')}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='result-box'>{t('ckd_risk_score').format(score=risk_score):.2f}</div>",
                            unsafe_allow_html=True)

                # Risk gauge
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=risk_score,
                    title={'text': t('risk_score', lang), 'font': {'color': '#FFD700'}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickcolor': '#FFF', 'tickfont': {'color': '#FFF'}},
                        'bar': {'color': "#1E3A8A"},
                        'steps': [
                            {'range': [0, 33], 'color': "#4ADE80"},
                            {'range': [33, 66], 'color': "#FACC15"},
                            {'range': [66, 100], 'color': "#EF4444"}
                        ],
                        'threshold': {
                            'line': {'color': "#FFD700", 'width': 4},
                            'thickness': 0.75,
                            'value': risk_score
                        }
                    }
                ))
                fig_gauge.update_layout(
                    paper_bgcolor="#0E1117",
                    font={'color': "#FFF"},
                    margin=dict(l=20, r=20, t=50, b=20)
                )
                st.plotly_chart(fig_gauge, use_container_width=True)

                # SHAP bar plot
                st.markdown(f"<div class='subheader'>{t('key_biomarkers')}</div>", unsafe_allow_html=True)
                shap_fig = plt.figure(figsize=(12, 8))
                sns.barplot(data=important_biomarkers.head(20), x='SHAP_Mean_Abs', y=t('feature'), hue=t('omic'),
                            palette='husl')
                plt.title(t('shap_top_features'), fontsize=16, color='#FFF')
                plt.xlabel(t('shap_mean_value'), fontsize=12, color='#FFF')
                plt.ylabel(t('feature'), fontsize=12, color='#FFF')
                plt.legend(title=t('omics'), facecolor='#1C2526', edgecolor='#FFF', labelcolor='#FFF')
                plt.gca().set_facecolor('#0E1117')
                plt.gca().tick_params(colors='#FFF')
                plt.gcf().set_facecolor('#0E1117')
                st.pyplot(shap_fig)

                # Display saved SHAP plot
                if os.path.exists(shap_plot_path):
                    st.image(shap_plot_path, caption=t('shap_caption'), use_column_width=True)

                # Recommendations
                st.markdown(f"<div class='subheader'>{t('recommendations')}</div>", unsafe_allow_html=True)
                for rec in recommendations:
                    st.markdown(f"<div class='recommendation'>{rec}</div>", unsafe_allow_html=True)

                # Download results
                st.markdown(f"<div class='subheader'>{t('download_results')}</div>", unsafe_allow_html=True)
                col1, col2 = st.columns(2)
                with col1:
                    csv = important_biomarkers.to_csv(index=False)
                    b64 = base64.b64encode(csv.encode()).decode()
                    href = f'<a href="data:file/csv;base64,{b64}" download="important_biomarkers.csv" class="download-btn">{t('download_biomarkers')}</a>'
                    st.markdown(href, unsafe_allow_html=True)
                with col2:
                    with open(pdf_path, "rb") as f:
                        pdf_data = f.read()
                    b64_pdf = base64.b64encode(pdf_data).decode()
                    href_pdf = f'<a href="data:application/pdf;base64,{b64_pdf}" download="ckd_risk_report.pdf" class="download-btn">{t('download_report')}</a>'
                    st.markdown(href_pdf, unsafe_allow_html=True)

            except Exception as e:
                st.error(t("file_process_error").format(error=str(e)))
                logger.error(f"Prediction error: {str(e)}")
        else:
            st.warning(t("upload_warning"))

# Chatbot Assistant Page
elif page == t("chatbot"):
    st.markdown(f"<div class='title animate__fadeIn'>{t('chatbot_title')}</div>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class='content animate__fadeIn'>
        <p>{t('chatbot_intro')}</p>
    </div>
    """, unsafe_allow_html=True)

    # Chat interface
    st.markdown(f"<div class='section-header'>{t('chat_with_assistant')}</div>", unsafe_allow_html=True)
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_history:
            if message['role'] == 'user':
                st.markdown(f"<div class='chat-bubble user'>{message['content']}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='chat-bubble bot'>{message['content']}</div>", unsafe_allow_html=True)

    # User input
    user_input = st.text_input(t("type_message"), key="chat_input")
    if st.button(t("send"), key="send_chat"):
        if user_input:
            # Add user message to history
            st.session_state.chat_history.append({'role': 'user', 'content': user_input})

            # Check if user is asking about results
            if st.session_state.results and any(
                    keyword in user_input.lower() for keyword in t('chat_risk_keywords').split(',')):
                risk_score = st.session_state.results['umap_df'][t('risk_score')].iloc[0]
                biomarkers = st.session_state.results['important_biomarkers'][t('feature')].head(5).tolist()
                context = t('chat_risk_response').format(score=risk_score, biomarkers=', '.join(biomarkers))
                response = send_to_rasa(f"{context} {user_input}", lang)
            else:
                # Generate recommendation based on intent
                intent = None
                if any(keyword in user_input.lower() for keyword in t('chat_medication_keywords').split(',')):
                    intent = "medication"
                elif any(keyword in user_input.lower() for keyword in t('chat_diet_keywords').split(',')):
                    intent = "diet"
                elif any(keyword in user_input.lower() for keyword in t('chat_exercise_keywords').split(',')):
                    intent = "exercise"

                if intent and st.session_state.results:
                    risk_score = st.session_state.results['umap_df'][t('risk_score')].iloc[0]
                    biomarkers = st.session_state.results['important_biomarkers'][t('feature')].head(5).tolist()
                    prompt = formulate_prompt(intent, risk_score, biomarkers)
                    response = generate_recommendation(prompt)
                    if validate_recommendation(response):
                        response = translate_recommendation(response, lang)
                    else:
                        response = t(f"{intent}_error", lang)
                else:
                    response = send_to_rasa(user_input, lang)

            # Add bot response to history
            st.session_state.chat_history.append({'role': 'bot', 'content': response})

            # Rerender chat
            st.experimental_rerun()

# Dashboard Page
elif page == t("dashboard"):
    st.markdown(f"<div class='title animate__fadeIn'>{t('dashboard_title')}</div>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class='content animate__fadeIn'>
        <p>{t('dashboard_intro')}</p>
    </div>
    """, unsafe_allow_html=True)

    try:
        session = Session()
        patients = session.query(Patient).all()
        if patients:
            patient_data = [{
                t('patient_id'): p.patient_id,
                t('risk_score'): p.risk_score,
                t('biomarkers'): p.biomarkers,
                t('date'): p.created_at.strftime('%Y-%m-%d %H:%M:%S')
            } for p in patients]
            df_patients = pd.DataFrame(patient_data)

            # Filters
            st.markdown(f"<div class='subheader'>{t('filters')}</div>", unsafe_allow_html=True)
            risk_threshold = st.slider(t("risk_threshold"), 0, 100, 50)
            filtered_df = df_patients[df_patients[t('risk_score')] >= risk_threshold]

            # Display table
            st.dataframe(filtered_df, use_container_width=True)

            # Plot
            fig = px.scatter(
                filtered_df,
                x=t('risk_score'),
                y=t('patient_id'),
                color=t('risk_score'),
                size=t('risk_score'),
                title=t('risk_distribution'),
                color_continuous_scale='Viridis'
            )
            fig.update_layout(
                paper_bgcolor="#0E1117",
                plot_bgcolor="#1C2526",
                font={'color': "#FFF"}
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(t("no_patients"))
        session.close()
    except Exception as e:
        st.error(t("dashboard_error").format(error=str(e)))
        logger.error(f"Dashboard error: {str(e)}")

# Logs Page
elif page == t("logs"):
    st.markdown(f"<div class='title animate__fadeIn'>{t('logs_title')}</div>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class='content animate__fadeIn'>
        <p>{t('logs_intro')}</p>
    </div>
    """, unsafe_allow_html=True)

    try:
        with open("logs/omics_predictor.log", "r") as f:
            logs = f.readlines()
        log_text = "".join(logs[-50:])
        st.text_area(t("recent_logs"), log_text, height=400)
    except FileNotFoundError:
        st.warning(t("logs_not_found"))
        logger.warning("Log file not found")

# Clean up temporary files
for omic, path in st.session_state.uploaded_files.items():
    if os.path.exists(path):
        os.remove(path)

# Terminate Rasa server on app close
if rasa_process:
    rasa_process.terminate()

st.markdown(f"""
<div class='footer'>
    {t('footer')}
</div>
""", unsafe_allow_html=True)