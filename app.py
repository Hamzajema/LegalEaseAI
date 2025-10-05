"""
Flask Backend API for Tunisia Legal Q&A System with Intelligent PDF Filling
"""

from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
import sys
import os
from pathlib import Path
import tempfile
import json
import re
from threading import Thread

def start_qa_system():
    initialize_qa_system()
    print("✅ QA system initialized!")
try:
    from TunisiaLegalQAa import TunisiaLegalQA
except ImportError:
    print("Error: Could not import TunisiaLegalQA from TunisiaLegalQAa.py")
    sys.exit(1)

try:
    from remplir_pdf2 import process_pdf_with_ai, call_groq_ai
except ImportError:
    print("Warning: Could not import from remplir_pdf2.py")
    process_pdf_with_ai = None
    call_groq_ai = None

app = Flask(__name__)
CORS(app)

qa_system = None
PDF_LEGALS_DIR = "./pdfs_legals"

# Conversation history per session
conversation_history = {}
@app.route('/')
def serve_html():
    # Serve the index.html from the static folder
    return send_from_directory(app.static_folder, 'index.html')

def initialize_qa_system():
    """Initialize the Q&A system"""
    global qa_system
    try:
        qa_system = TunisiaLegalQA(
            pdf_directory="./tunisia_legal_pdfs",
            model_name="phi-2",
            use_gpu=True
        )
        
        print("Loading processed documents...")
        qa_system.process_pdfs()
        
        print(f"✅ System initialized with {len(qa_system.documents)} document chunks")
        return True
    except Exception as e:
        print(f"❌ Error initializing system: {e}")
        return False

def validate_legal_answer(question, answer, sources, max_retries=5):
    """
    Supervisor LLM that validates if the legal answer is correct and well-formatted.
    Returns: (is_valid, feedback, improved_answer)
    """
    
    if not call_groq_ai:
        return True, "Validation unavailable", answer
    
    # Build sources context
    sources_text = "\n".join([
        f"Source {i+1}: {s.get('source', 'Unknown')} - Page {s.get('page', 'N/A')}\n{s.get('content', '')[:300]}..."
        for i, s in enumerate(sources)
    ])
    
    validation_prompt = f"""Tu es un superviseur juridique expert en droit tunisien. Ton rôle est de VALIDER si une réponse juridique est correcte, complète et bien présentée.

QUESTION POSÉE:
{question}

RÉPONSE FOURNIE PAR LE SYSTÈME:
{answer}

SOURCES UTILISÉES:
{sources_text}

CRITÈRES DE VALIDATION (TRÈS IMPORTANTS):
1. ✅ EXACTITUDE JURIDIQUE: La réponse est-elle juridiquement correcte selon les sources?
2. ✅ COMPLÉTUDE: La réponse répond-elle complètement à la question?
3. ✅ CLARTÉ: La réponse est-elle claire et bien structurée?
4. ✅ PERTINENCE: La réponse reste-t-elle focalisée sur la question sans dévier?
5. ✅ SOURCES: La réponse est-elle cohérente avec les sources fournies?

CRITÈRES D'INVALIDATION (REJETER LA RÉPONSE SI):
❌ Informations juridiques incorrectes ou contradictoires
❌ Répond à une autre question que celle posée
❌ Contient des hallucinations (invente des lois/articles qui n'existent pas)
❌ Mentionne "Solution 0:" ou autres artefacts techniques
❌ Réponse vide, trop courte (<30 mots) ou incompréhensible
❌ Ne correspond pas du tout aux sources fournies
❌ Donne des conseils dangereux ou illégaux

INSTRUCTIONS:
Si la réponse est VALIDE (score >= 8/10): Retourne is_valid=true
Si la réponse est INVALIDE (score < 8/10): Retourne is_valid=false et fournis une version AMÉLIORÉE

Retourne UNIQUEMENT ce JSON:
{{
    "is_valid": true/false,
    "score": 0-10,
    "feedback": "explication détaillée de la validation",
    "issues": ["problème1", "problème2", ...],
    "improved_answer": "version améliorée si invalide, sinon null"
}}

IMPORTANT: Si invalide, l'improved_answer doit être une réponse complète et correcte basée sur les sources."""

    try:
        response = call_groq_ai(validation_prompt)
        
        # Clean and parse response
        cleaned = response.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        
        validation_result = json.loads(cleaned.strip())
        
        is_valid = validation_result.get('is_valid', False)
        score = validation_result.get('score', 0)
        feedback = validation_result.get('feedback', 'No feedback')
        improved = validation_result.get('improved_answer')
        
        print(f"\n   🔍 VALIDATION RESULT:")
        print(f"   - Valid: {is_valid}")
        print(f"   - Score: {score}/10")
        print(f"   - Feedback: {feedback[:100]}...")
        
        return is_valid, feedback, improved if improved else answer
        
    except Exception as e:
        print(f"   ⚠️ Validation error: {e}")
        # If validation fails, assume answer is valid to not block the system
        return True, "Validation error", answer

def get_validated_answer(question, top_k=3, language='en', max_attempts=5):
    """
    Get answer from TunisiaLegalQA with supervisor validation.
    Retries up to max_attempts times if answer is invalid.
    """
    
    print(f"\n🔄 Starting validated Q&A (max {max_attempts} attempts)...")
    
    for attempt in range(1, max_attempts + 1):
        print(f"\n{'='*60}")
        print(f"Attempt {attempt}/{max_attempts}")
        print(f"{'='*60}")
        
        # Get answer from main Q&A system
        result = qa_system.answer_question(
            question=question,
            top_k=top_k,
            use_llm=True,
            language=language
        )
        
        answer = result['answer']
        sources = result.get('sources', [])
        
        # Clean common artifacts
        if answer.startswith("Solution 0:"):
            answer = answer.replace("Solution 0:", "").strip()
        answer = answer.replace("[/inst]", "").strip()
        
        print(f"   📝 Generated answer length: {len(answer)} chars")
        
        # Validate the answer
        is_valid, feedback, improved_answer = validate_legal_answer(
            question, 
            answer, 
            sources,
            max_attempts
        )
        
        if is_valid:
            print(f"   ✅ Answer VALIDATED on attempt {attempt}")
            return {
                'question': question,
                'answer': improved_answer,
                'sources': sources,
                'confidence': result.get('confidence', 0.7),
                'attempts': attempt,
                'validation_feedback': feedback
            }
        else:
            print(f"   ❌ Answer REJECTED on attempt {attempt}")
            print(f"   📋 Issues: {feedback}")
            
            if attempt < max_attempts:
                print(f"   🔄 Retrying with improved prompt...")
                # Use the improved answer for next iteration if available
                if improved_answer and improved_answer != answer:
                    print(f"   💡 Using supervisor's improved version")
    
    # If all attempts fail, return the last improved answer with warning
    print(f"\n   ⚠️ Max attempts reached. Returning best available answer.")
    return {
        'question': question,
        'answer': improved_answer if improved_answer else answer,
        'sources': sources,
        'confidence': 0.5,  # Lower confidence since validation failed
        'attempts': max_attempts,
        'validation_feedback': f"Warning: Answer could not be fully validated after {max_attempts} attempts. {feedback}",
        'validation_failed': True
    }

def analyze_user_intent(message, conversation_context=""):
    """Use LLM to analyze if user wants to fill a form and extract information"""
    
    if not call_groq_ai:
        return None
    
    # List available PDFs
    pdf_files = []
    if os.path.exists(PDF_LEGALS_DIR):
        pdf_files = [f for f in os.listdir(PDF_LEGALS_DIR) if f.endswith('.pdf')]
    
    pdf_list_str = "\n".join([f"- {pdf}" for pdf in pdf_files])
    
    prompt = f"""Tu es un assistant juridique intelligent pour la Tunisie.

PDFS DISPONIBLES:
{pdf_list_str}

HISTORIQUE DE CONVERSATION:
{conversation_context}

MESSAGE UTILISATEUR:
{message}

ANALYSE REQUISE:
1. Est-ce que l'utilisateur demande de remplir un formulaire/document légal? (oui/non)
2. Si oui, quel PDF correspond le mieux à sa demande?
3. Extrais TOUTES les informations fournies par l'utilisateur

RÈGLES DE SÉLECTION PDF:
- "SARL" (Société à Responsabilité Limitée) → demande_immatriculation_SARL.pdf ou Statut_Type_SARL.pdf ou Modéle STATUTS_SARL.pdf
- "SUARL" (Société Unipersonnelle) → demande_immatriculation_SUARL.pdf ou Statut_Type_SUARL.pdf ou Modéle *STATUS* SUARL.pdf
- "SA" (Société Anonyme) → Modèle *STATUTS* SA_0.pdf
- "statuts" → fichiers STATUTS/STATUS correspondants
- "immatriculation" → fichiers demande_immatriculation
- "PV" ou "procès-verbal" → Modéle PV_AGC.pdf ou Modèle PV_Conseil d'administration.pdf
- "liasse" → Liasse unique (Français).pdf

EXTRACTION D'INFORMATIONS:
Cherche et extrait:
- Nom de l'entreprise/société
- Type de société (SARL, SUARL, SA)
- Capital social
- Adresse du siège social
- Noms des fondateurs/associés
- Prénoms
- Durée de la société
- Objet social
- Activité
- Numéro de téléphone
- Email
- Dates importantes
- Numéro de compte bancaire
- Banque
- Toute autre information pertinente

Retourne UNIQUEMENT un JSON avec cette structure:
{{
    "wants_pdf": true/false,
    "pdf_filename": "nom_exact_du_fichier.pdf" ou null,
    "confidence": 0.0-1.0,
    "reason": "explication du choix",
    "extracted_data": {{
        "entreprise": "...",
        "capital": "...",
        "adresse": "...",
        ...tous les champs trouvés...
    }}
}}

Si aucun PDF ne correspond ou pas assez d'infos: {{"wants_pdf": false}}"""

    try:
        response = call_groq_ai(prompt)
        
        # Clean response
        cleaned = response.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        
        result = json.loads(cleaned.strip())
        return result
    
    except Exception as e:
        print(f"Error analyzing intent: {e}")
        return None

def extract_additional_info_from_conversation(conversation_context):
    """Extract any additional information from conversation history"""
    
    if not call_groq_ai or not conversation_context:
        return {}
    
    prompt = f"""Analyse cette conversation et extrait TOUTES les informations utiles pour remplir un formulaire légal tunisien.

CONVERSATION:
{conversation_context}

Extrait et retourne un JSON avec toutes les informations trouvées:
{{
    "nom": "...",
    "prenom": "...",
    "entreprise": "...",
    "capital": "...",
    "adresse": "...",
    "telephone": "...",
    "email": "...",
    "objet_social": "...",
    "duree": "...",
    ...autres champs pertinents...
}}

Retourne UNIQUEMENT le JSON, rien d'autre."""

    try:
        response = call_groq_ai(prompt)
        cleaned = response.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        
        return json.loads(cleaned.strip())
    except:
        return {}

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    if qa_system and len(qa_system.documents) > 0:
        return jsonify({
            'status': 'healthy',
            'documents_loaded': len(qa_system.documents),
            'message': 'System is ready'
        }), 200
    else:
        return jsonify({
            'status': 'initializing',
            'message': 'System is still loading'
        }), 503

@app.route('/ask', methods=['POST'])
def ask_question():
    """Main endpoint - handles both Q&A and intelligent PDF filling"""
    try:
        if not qa_system:
            return jsonify({
                'error': 'System not initialized'
            }), 503
        
        data = request.get_json()
        
        if not data or 'question' not in data:
            return jsonify({
                'error': 'Missing question'
            }), 400
        
        question = data['question']
        session_id = data.get('session_id', 'default')
        language = data.get('language', 'en')
        
        # Update conversation history
        if session_id not in conversation_history:
            conversation_history[session_id] = []
        
        conversation_history[session_id].append({
            'role': 'user',
            'content': question
        })
        
        # Build conversation context
        context = "\n".join([
            f"{msg['role']}: {msg['content']}" 
            for msg in conversation_history[session_id][-5:]  # Last 5 messages
        ])
        
        print(f"\n{'='*60}")
        print(f"Processing: {question[:100]}...")
        
        # STEP 1: Analyze if user wants to fill a PDF
        intent_analysis = analyze_user_intent(question, context)
        
        if intent_analysis and intent_analysis.get('wants_pdf') and intent_analysis.get('pdf_filename'):
            print(f"📄 PDF Request Detected!")
            print(f"   PDF: {intent_analysis['pdf_filename']}")
            print(f"   Confidence: {intent_analysis.get('confidence', 0)}")
            
            # STEP 2: Merge extracted data with conversation context
            user_data = intent_analysis.get('extracted_data', {})
            additional_data = extract_additional_info_from_conversation(context)
            user_data.update(additional_data)
            
            print(f"   Extracted fields: {list(user_data.keys())}")
            
            # STEP 3: Check if PDF exists
            pdf_path = os.path.join(PDF_LEGALS_DIR, intent_analysis['pdf_filename'])
            
            if not os.path.exists(pdf_path):
                # Try to find similar PDF
                similar_pdf = find_similar_pdf(intent_analysis['pdf_filename'])
                if similar_pdf:
                    pdf_path = os.path.join(PDF_LEGALS_DIR, similar_pdf)
                    intent_analysis['pdf_filename'] = similar_pdf
                else:
                    answer = f"Je comprends que vous voulez remplir un formulaire, mais je n'ai pas trouvé le PDF '{intent_analysis['pdf_filename']}'. PDFs disponibles: {', '.join(os.listdir(PDF_LEGALS_DIR))}"
                    conversation_history[session_id].append({
                        'role': 'assistant',
                        'content': answer
                    })
                    return jsonify({
                        'question': question,
                        'answer': answer,
                        'type': 'error',
                        'sources': [],
                        'confidence': 0
                    }), 200
            
            # STEP 4: Fill the PDF
            if process_pdf_with_ai:
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                        output_path = tmp_file.name
                    
                    print(f"   Filling PDF...")
                    result = process_pdf_with_ai(
                        pdf_path=pdf_path,
                        user_data=user_data,
                        output_path=output_path,
                        interactive=False
                    )
                    
                    if result['success']:
                        # Read PDF as base64
                        import base64
                        with open(output_path, 'rb') as f:
                            pdf_base64 = base64.b64encode(f.read()).decode('utf-8')
                        
                        answer = f"✅ J'ai rempli le formulaire '{intent_analysis['pdf_filename']}' avec vos informations. {len(result['filled_fields'])} champs ont été complétés."
                        
                        if result.get('missing_fields'):
                            answer += f"\n\n⚠️ Informations manquantes: {', '.join(result['missing_fields'])}"
                        
                        conversation_history[session_id].append({
                            'role': 'assistant',
                            'content': answer
                        })
                        
                        return jsonify({
                            'question': question,
                            'answer': answer,
                            'type': 'pdf_generated',
                            'pdf_filename': intent_analysis['pdf_filename'],
                            'pdf_data': pdf_base64,
                            'filled_fields': result['filled_fields'],
                            'missing_fields': result.get('missing_fields', []),
                            'confidence': intent_analysis.get('confidence', 0.8)
                        }), 200
                    else:
                        answer = f"❌ Erreur lors du remplissage du PDF. Raison: {result.get('error', 'Inconnue')}"
                        
                except Exception as e:
                    print(f"❌ PDF Filling Error: {e}")
                    import traceback
                    traceback.print_exc()
                    answer = f"❌ Erreur lors du traitement du PDF: {str(e)}"
            else:
                answer = "Le module de remplissage PDF n'est pas disponible."
            
            conversation_history[session_id].append({
                'role': 'assistant',
                'content': answer
            })
            
            return jsonify({
                'question': question,
                'answer': answer,
                'type': 'error',
                'sources': [],
                'confidence': 0
            }), 200
        
        # STEP 5: Normal Q&A if no PDF request
        print("💬 Regular Q&A Request")
        
        result = qa_system.answer_question(
            question=question,
            top_k=data.get('top_k', 3),
            use_llm=data.get('use_llm', True),
            language=language
        )
        
        answer = result['answer']
        if answer.startswith("Solution 0:"):
            answer = answer.replace("Solution 0:", "").strip()
        answer = answer.replace("[/inst]", "").strip()
        
        conversation_history[session_id].append({
            'role': 'assistant',
            'content': answer
        })
        
        return jsonify({
            'question': result['question'],
            'answer': answer,
            'type': 'qa',
            'sources': result['sources'],
            'confidence': result['confidence'],
            'language': language
        }), 200
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': 'Processing error',
            'message': str(e)
        }), 500

def find_similar_pdf(filename):
    """Find similar PDF if exact match not found"""
    if not os.path.exists(PDF_LEGALS_DIR):
        return None
    
    available = [f for f in os.listdir(PDF_LEGALS_DIR) if f.endswith('.pdf')]
    
    # Simple similarity matching
    filename_lower = filename.lower()
    for pdf in available:
        if pdf.lower() == filename_lower:
            return pdf
        # Check if key terms match
        if any(term in pdf.lower() for term in filename_lower.split('_')):
            return pdf
    
    return None

@app.route('/stats', methods=['GET'])
def get_stats():
    """Get system statistics"""
    try:
        if not qa_system:
            return jsonify({'error': 'System not initialized'}), 503
        
        sources = set()
        for meta in qa_system.metadata:
            sources.add(meta['source'])
        
        pdf_count = 0
        if os.path.exists(PDF_LEGALS_DIR):
            pdf_count = len([f for f in os.listdir(PDF_LEGALS_DIR) if f.endswith('.pdf')])
        
        stats = {
            'total_chunks': len(qa_system.documents),
            'total_sources': len(sources),
            'model_name': qa_system.model_name,
            'device': qa_system.device,
            'pdf_forms_available': pdf_count,
            'pdf_filling_enabled': process_pdf_with_ai is not None,
            'active_sessions': len(conversation_history)
        }
        
        return jsonify(stats), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("=" * 80)
    print("Tunisia Legal Q&A System - Intelligent PDF Filling")
    print("=" * 80)
    
    if os.path.exists(PDF_LEGALS_DIR):
        pdf_count = len([f for f in os.listdir(PDF_LEGALS_DIR) if f.endswith('.pdf')])
        print(f"✅ Found {pdf_count} PDF forms")
    
    if initialize_qa_system():
        print("\n🚀 Starting Flask server with intelligent PDF detection...")
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    else:
        sys.exit(1)