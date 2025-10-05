import requests
import json
from typing import Dict, Any
import PyPDF2
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from io import BytesIO
import fitz  # PyMuPDF pour meilleure extraction
from pathlib import Path

def call_groq_ai(prompt: str) -> str:
    """Appelle l'API Groq pour générer du contenu"""
    try:
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": "Bearer ",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "llama-3.3-70b-versatile",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1024,
            "temperature": 0.7
        }
        response = requests.post(url, headers=headers, json=payload)
        data = response.json()
        
        if response.status_code != 200:
            print("Groq API Error Response:", data)
            return f"Error: {data.get('error', {}).get('message', 'Unknown error')}"
        
        return data.get("choices", [{}])[0].get("message", {}).get("content", "").strip() or "No response"
    except Exception as e:
        print("Groq API Error:", e)
        return "Error contacting AI."


def extract_pdf_structure(pdf_path: str) -> Dict[str, Any]:
    """Extrait la structure et le contenu du PDF"""
    try:
        # Utiliser PyMuPDF pour une meilleure extraction
        doc = fitz.open(pdf_path)
        
        pdf_data = {
            "pages": [],
            "text_content": "",
            "form_fields": [],
            "metadata": {}
        }
        
        # Extraire les métadonnées
        pdf_data["metadata"] = doc.metadata
        
        # Extraire le contenu de chaque page
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # Extraire le texte
            text = page.get_text()
            pdf_data["text_content"] += f"\n--- Page {page_num + 1} ---\n{text}"
            
            # Extraire les widgets (champs de formulaire)
            widgets = page.widgets()
            for widget in widgets:
                field_info = {
                    "page": page_num + 1,
                    "name": widget.field_name or f"field_{page_num}_{widget.rect}",
                    "type": widget.field_type_string,
                    "value": widget.field_value,
                    "rect": list(widget.rect),
                }
                pdf_data["form_fields"].append(field_info)
            
            # Stocker les informations de la page
            pdf_data["pages"].append({
                "number": page_num + 1,
                "width": page.rect.width,
                "height": page.rect.height,
                "text": text
            })
        
        doc.close()
        return pdf_data
        
    except Exception as e:
        print(f"Erreur lors de l'extraction du PDF: {e}")
        # Fallback vers PyPDF2
        return extract_pdf_with_pypdf2(pdf_path)


def extract_pdf_with_pypdf2(pdf_path: str) -> Dict[str, Any]:
    """Extraction de secours avec PyPDF2"""
    pdf_data = {
        "pages": [],
        "text_content": "",
        "form_fields": [],
        "metadata": {}
    }
    
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            pdf_data["metadata"] = reader.metadata or {}
            
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                pdf_data["text_content"] += f"\n--- Page {page_num + 1} ---\n{text}"
                pdf_data["pages"].append({
                    "number": page_num + 1,
                    "text": text
                })
                
                # Essayer d'extraire les champs de formulaire
                if '/Annots' in page:
                    for annot in page['/Annots']:
                        obj = annot.get_object()
                        if obj.get('/Subtype') == '/Widget':
                            field_name = obj.get('/T', 'unknown')
                            field_value = obj.get('/V', '')
                            pdf_data["form_fields"].append({
                                "page": page_num + 1,
                                "name": field_name,
                                "value": field_value
                            })
    except Exception as e:
        print(f"Erreur avec PyPDF2: {e}")
    
    return pdf_data


def prepare_ai_prompt(pdf_data: Dict[str, Any], user_data: Dict[str, Any]) -> str:
    """Prépare le prompt pour l'API Groq"""
    print("📝 Contenu du PDF envoyé à l'IA (extrait):")
    print(pdf_data['text_content'][:500])
    print("...\n")
    
    # Construire une description des champs du PDF
    fields_description = ""
    if pdf_data["form_fields"]:
        fields_description = "Champs du formulaire détectés:\n"
        for field in pdf_data["form_fields"]:
            fields_description += f"- {field['name']} (Page {field['page']})\n"
    
    prompt = f"""Tu es un assistant expert qui remplit des formulaires PDF légaux tunisiens.

DONNÉES UTILISATEUR DISPONIBLES:
{json.dumps(user_data, indent=2, ensure_ascii=False)}

STRUCTURE DU PDF:
{fields_description}

CONTENU DU PDF (avec les champs vides marqués par des pointillés …):
{pdf_data['text_content'][:4000]}

INSTRUCTIONS CRITIQUES:
1. Analyse TOUS les endroits avec des pointillés (…) ou espaces vides dans le PDF
2. Pour CHAQUE champ vide identifié, cherche dans les données utilisateur la valeur appropriée
3. Si les données ne contiennent pas exactement ce qui est demandé, EXTRAIT l'information pertinente du texte fourni
4. Les champs à remplir incluent typiquement:
   - Nom de la société
   - Capital social (montant en DT)
   - Adresse du siège social
   - Noms des associés/fondateurs
   - Durée de la société (en années)
   - Nombre et valeur des parts sociales
   - Informations bancaires (banque, agence, numéro de compte)
   - Dates et signatures

5. IMPORTANT: Tu dois identifier et remplir TOUS les champs vides, pas seulement quelques-uns

Retourne un JSON avec cette structure exacte:
{{
    "filled_fields": [
        {{
            "field_name": "description exacte du champ (ex: 'au capital de', 'Siège Social', 'Durée')",
            "page": numéro de page,
            "value": "valeur extraite ou calculée",
            "source": "explication d'où vient cette valeur"
        }}
    ],
    "missing_fields": ["seulement si vraiment impossible de trouver l'info"]
}}

RETOURNE UNIQUEMENT LE JSON, sans texte avant ou après."""

    return prompt


def fill_pdf_with_ai_data(
    pdf_path: str,
    output_path: str,
    filled_data: Dict[str, Any],
    pdf_structure: Dict[str, Any]
) -> str:
    """Crée un nouveau PDF avec les données remplies"""
    
    try:
        # Ouvrir le PDF original
        doc = fitz.open(pdf_path)
        
        has_form_fields = len(pdf_structure.get("form_fields", [])) > 0
        fields_filled = 0
        used_dots = []  # Garder trace des pointillés déjà utilisés
        
        if has_form_fields:
            # Méthode 1: Remplir les champs de formulaire interactifs
            for field_info in filled_data.get("filled_fields", []):
                page_num = field_info.get("page", 1) - 1
                if page_num < 0 or page_num >= len(doc):
                    continue
                    
                page = doc[page_num]
                
                # Chercher le widget correspondant
                for widget in page.widgets():
                    if widget.field_name == field_info["field_name"]:
                        widget.field_value = str(field_info["value"])
                        widget.update()
                        fields_filled += 1
                        print(f"✓ Champ rempli: {field_info['field_name']}")
        else:
            # Méthode 2: Ajouter du texte directement sur le PDF (pas de champs interactifs)
            print("⚠️ Aucun champ de formulaire détecté. Ajout de texte sur le PDF...\n")
            
            # Extraire tous les pointillés sur toutes les pages
            all_dots_by_page = {}
            for p_num in range(len(doc)):
                p = doc[p_num]
                dots_list = []
                
                # Chercher différents patterns de pointillés
                for pattern in ["…………………………", "…………………", "…………", "……", "…", ".....", "....."]:
                    found = p.search_for(pattern)
                    for rect in found:
                        # Éviter les doublons (même zone)
                        is_duplicate = any(
                            abs(rect.x0 - d.x0) < 10 and abs(rect.y0 - d.y0) < 2 
                            for d in dots_list
                        )
                        if not is_duplicate:
                            dots_list.append(rect)
                
                # Trier par position (haut en bas, gauche à droite)
                dots_list.sort(key=lambda r: (round(r.y0), r.x0))
                all_dots_by_page[p_num] = dots_list
                print(f"📍 Page {p_num + 1}: {len(dots_list)} zones de pointillés détectées")
            
            print()
            
            for field_info in filled_data.get("filled_fields", []):
                page_num = field_info.get("page", 1) - 1
                if page_num < 0 or page_num >= len(doc):
                    continue
                
                page = doc[page_num]
                field_name = field_info["field_name"]
                value = str(field_info["value"])
                
                print(f"🔍 Recherche position pour: {field_name[:60]}...")
                
                # Stratégie 1: Chercher le label du champ et les pointillés à proximité
                text_instances = page.search_for(field_name)
                
                if not text_instances:
                    # Chercher par mots-clés
                    keywords = [k for k in field_name.split() if len(k) > 3]
                    for keyword in keywords:
                        text_instances = page.search_for(keyword)
                        if text_instances:
                            print(f"   → Trouvé via mot-clé: {keyword}")
                            break
                
                position_found = False
                
                if text_instances:
                    label_rect = text_instances[0]
                    print(f"   → Label trouvé à: ({label_rect.x0:.1f}, {label_rect.y0:.1f})")
                    
                    # Chercher les pointillés proches (même ligne, à droite du label)
                    available_dots = all_dots_by_page.get(page_num, [])
                    
                    for dots_rect in available_dots:
                        # Les pointillés doivent être sur la même ligne et après le label
                        same_line = abs(dots_rect.y0 - label_rect.y0) < 10
                        after_label = dots_rect.x0 > label_rect.x1
                        not_used = not any(
                            abs(dots_rect.x0 - ux) < 10 and abs(dots_rect.y0 - uy) < 10
                            for ux, uy in used_dots
                        )
                        
                        if same_line and after_label and not_used:
                            # Position trouvée !
                            # Calculer la position Y pour un bon alignement vertical
                            baseline_y = dots_rect.y1 - 2  # 2 pixels au-dessus du bas des pointillés
                            
                            # Effacer les pointillés
                            clear_rect = fitz.Rect(
                                dots_rect.x0 - 1,
                                dots_rect.y0 - 1,
                                dots_rect.x0 + 400,  # Large pour couvrir long texte
                                dots_rect.y1 + 1
                            )
                            page.draw_rect(clear_rect, color=(1, 1, 1), fill=(1, 1, 1))
                            
                            # Insérer le texte
                            page.insert_text(
                                (dots_rect.x0 + 1, baseline_y),
                                value,
                                fontsize=9,
                                color=(0, 0, 0),
                                fontname="helv"
                            )
                            
                            used_dots.append((dots_rect.x0, dots_rect.y0))
                            fields_filled += 1
                            position_found = True
                            print(f"   ✅ Texte ajouté à ({dots_rect.x0:.1f}, {baseline_y:.1f}): {value}\n")
                            break
                
                # Stratégie 2: Si aucun label trouvé, utiliser les pointillés non utilisés dans l'ordre
                if not position_found:
                    available_dots = all_dots_by_page.get(page_num, [])
                    
                    for dots_rect in available_dots:
                        not_used = not any(
                            abs(dots_rect.x0 - ux) < 10 and abs(dots_rect.y0 - uy) < 10
                            for ux, uy in used_dots
                        )
                        
                        if not_used:
                            baseline_y = dots_rect.y1 - 2
                            
                            # Effacer les pointillés
                            clear_rect = fitz.Rect(
                                dots_rect.x0 - 1,
                                dots_rect.y0 - 1,
                                dots_rect.x0 + 400,
                                dots_rect.y1 + 1
                            )
                            page.draw_rect(clear_rect, color=(1, 1, 1), fill=(1, 1, 1))
                            
                            # Insérer le texte
                            page.insert_text(
                                (dots_rect.x0 + 1, baseline_y),
                                value,
                                fontsize=9,
                                color=(0, 0, 0),
                                fontname="helv"
                            )
                            
                            used_dots.append((dots_rect.x0, dots_rect.y0))
                            fields_filled += 1
                            position_found = True
                            print(f"   ✅ Texte ajouté (pointillés séquentiels) à ({dots_rect.x0:.1f}, {baseline_y:.1f}): {value}\n")
                            break
                
                if not position_found:
                    print(f"   ⚠️ Aucune position disponible\n")
        
        print(f"\n{'='*60}")
        print(f"📊 Total: {fields_filled} champs remplis sur {len(filled_data.get('filled_fields', []))}")
        print(f"{'='*60}\n")
        
        # Sauvegarder le PDF modifié
        doc.save(output_path)
        doc.close()
        
        return output_path
        
    except Exception as e:
        print(f"❌ Erreur lors du remplissage du PDF: {e}")
        import traceback
        traceback.print_exc()
        # Fallback: créer un overlay avec ReportLab
        return create_pdf_overlay(pdf_path, output_path, filled_data, pdf_structure)


def create_pdf_overlay(
    pdf_path: str,
    output_path: str,
    filled_data: Dict[str, Any],
    pdf_structure: Dict[str, Any]
) -> str:
    """Crée un overlay sur le PDF existant avec ReportLab"""
    
    try:
        from PyPDF2 import PdfReader, PdfWriter
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter
        import io
        
        # Lire le PDF original
        reader = PdfReader(pdf_path)
        writer = PdfWriter()
        
        # Créer un dictionnaire des champs par page
        fields_by_page = {}
        for field in filled_data.get("filled_fields", []):
            page_num = field.get("page", 1) - 1
            if page_num not in fields_by_page:
                fields_by_page[page_num] = []
            fields_by_page[page_num].append(field)
        
        # Traiter chaque page
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            
            # S'il y a des champs à remplir sur cette page
            if page_num in fields_by_page:
                # Créer un overlay avec ReportLab
                packet = io.BytesIO()
                
                # Obtenir les dimensions de la page
                page_width = float(page.mediabox.width)
                page_height = float(page.mediabox.height)
                
                can = canvas.Canvas(packet, pagesize=(page_width, page_height))
                can.setFont("Helvetica", 10)
                
                # Ajouter les textes sur l'overlay
                for field in fields_by_page[page_num]:
                    field_name = field["field_name"]
                    value = str(field["value"])
                    
                    # Positions estimées (à ajuster selon votre PDF)
                    # Ces positions sont basées sur une analyse du contenu
                    x, y = estimate_field_position(field_name, page_height)
                    
                    if x and y:
                        can.drawString(x, y, value)
                        print(f"✓ Overlay ajouté: {field_name[:50]}... = {value}")
                
                can.save()
                
                # Fusionner l'overlay avec la page originale
                packet.seek(0)
                overlay_pdf = PdfReader(packet)
                page.merge_page(overlay_pdf.pages[0])
            
            writer.add_page(page)
        
        # Ajouter les métadonnées
        if pdf_structure.get("metadata"):
            writer.add_metadata(pdf_structure["metadata"])
        
        # Écrire le PDF final
        with open(output_path, 'wb') as output_file:
            writer.write(output_file)
        
        return output_path
        
    except Exception as e:
        print(f"Erreur lors de la création de l'overlay: {e}")
        import traceback
        traceback.print_exc()
        return None


def estimate_field_position(field_name: str, page_height: float) -> tuple:
    """Estime la position d'un champ basé sur son nom"""
    
    # Mapping des positions communes (à adapter selon votre PDF)
    positions = {
        "capital": (200, page_height - 100),
        "Siège Social": (200, page_height - 120),
        "soussign": (100, page_height - 180),
        "Dénomination": (200, page_height - 400),
        "Durée": (200, page_height - 450),
        "Capital social": (200, page_height - 500),
    }
    
    # Rechercher une correspondance partielle
    for key, pos in positions.items():
        if key.lower() in field_name.lower():
            return pos
    
    # Position par défaut
    return (150, page_height - 200)


def process_pdf_with_ai(
    pdf_path: str,
    user_data: Dict[str, Any],
    output_path: str = "filled_form.pdf"
) -> Dict[str, Any]:
    """Fonction principale pour traiter le PDF avec l'IA"""
    
    print("📄 Extraction de la structure du PDF...")
    pdf_structure = extract_pdf_structure(pdf_path)
    
    print(f"✓ {len(pdf_structure['pages'])} pages extraites")
    print(f"✓ {len(pdf_structure['form_fields'])} champs de formulaire détectés")
    
    print("\n🤖 Préparation du prompt pour l'IA...")
    prompt = prepare_ai_prompt(pdf_structure, user_data)
    
    print("🔄 Appel de l'API Groq...")
    ai_response = call_groq_ai(prompt)
    
    print("\n📝 Réponse de l'IA:")
    print(ai_response)
    
    # Parser la réponse JSON
    try:
        # Nettoyer la réponse (enlever les markdown code blocks si présents)
        cleaned_response = ai_response.strip()
        if cleaned_response.startswith("```json"):
            cleaned_response = cleaned_response[7:]
        if cleaned_response.startswith("```"):
            cleaned_response = cleaned_response[3:]
        if cleaned_response.endswith("```"):
            cleaned_response = cleaned_response[:-3]
        
        filled_data = json.loads(cleaned_response.strip())
    except json.JSONDecodeError as e:
        print(f"⚠️ Erreur de parsing JSON: {e}")
        filled_data = {"filled_fields": [], "missing_fields": []}
    
    print("\n📝 Création du PDF rempli...")
    result_path = fill_pdf_with_ai_data(pdf_path, output_path, filled_data, pdf_structure)
    
    if result_path:
        print(f"✅ PDF créé avec succès: {result_path}")
    else:
        print("❌ Erreur lors de la création du PDF")
    
    return {
        "success": result_path is not None,
        "output_path": result_path,
        "filled_fields": filled_data.get("filled_fields", []),
        "missing_fields": filled_data.get("missing_fields", []),
        "pdf_structure": pdf_structure
    }


# Exemple d'utilisation
if __name__ == "__main__":
    # Données utilisateur de votre application
    user_data = {
    "startupobjet": "La société d'expertise comptable et de commissariat aux comptes, Imaginaire Conseil SARL, au capital de 50 000 Dinars Tunisiens, a son siège social situé au 12 Rue de l'Innovation, 1002 Tunis, Tunisie.",
    "realisation":" Les soussignés, M. Hamza Jemaa et Mme Latifa Jaafari, ont établi les statuts de cette société à responsabilité limitée conformément aux lois et règlements applicables, notamment la loi 88-108 relative à l'exercice de la profession d'expert-comptable en Tunisie. La société a pour objet l'exercice de la profession d'expert-comptable, incluant la réalisation de missions de comptabilité, d'audit et de conseil aux entreprises. Elle peut réaliser toutes opérations compatibles avec son objet social, mais ne peut prendre de participations financières dans d'autres sociétés industrielles, commerciales, agricoles ou bancaires, ni dans des sociétés civiles, et ne doit être sous la dépendance, directe ou indirecte, d'aucune autre entité. La dénomination sociale de la société est Imaginaire Conseil SARL. Le siège social est fixé au 12 Rue de l'Innovation, 1002 Tunis, Tunisie. La durée de la société est fixée à 99 ans à compter de son immatriculation au registre du commerce, sauf dissolution anticipée ou prorogation décidée par les associés selon les règles de délibération en assemblée générale extraordinaire. Le capital social, fixé à 50 000 Dinars Tunisiens, est divisé en 500 parts sociales de 100 D.T chacune, attribuées aux associés proportionnellement à leurs apports : 250 parts à M. Hamza Jemaa et 250 parts à Mme Latifa Jaafari. Les apports en numéraire, intégralement libérés, ont été déposés au compte indisponible ouvert auprès de la Banque Centrale de Tunisie, agence Tunis-Centre, n° 123456789. Entre les associés, les parts sociales sont librement cessibles. Toute cession à des tiers nécessite le consentement de la majorité des associés représentant au moins les trois quarts du capital social et respecte la législation sur la détention des parts par les experts-comptables. Toute modification du capital, par augmentation ou réduction, sera décidée conformément aux statuts et à la loi, tout en respectant la répartition prévue initialement. En cas de décès ou d'incapacité d'un associé, la société continue avec les associés restants ou par l'agrément de nouveaux associés, en veillant à ce qu'aucun non-membre de l'Ordre des Experts Comptables ne détienne plus de 25 % du capital. La société tient un registre des dirigeants et membres du conseil de surveillance, ainsi qu'un registre des parts sociales avec l'identité des propriétaires. Les statuts enregistrés et toutes modifications sont communiqués à l'Ordre des Experts Comptables et disponibles pour les pouvoirs publics et tiers intéressés. M. Hamza Jemaa est nommé gérant pour une durée de trois ans renouvelable. Il dispose des pouvoirs les plus étendus pour agir au nom de la société, sous réserve des pouvoirs légaux attribués aux associés. Le gérant perçoit une rémunération mensuelle de 2 500 D.T, et ses frais de représentation et de déplacement sont remboursés sur présentation de justificatifs. Les décisions des associés sont qualifiées d'ordinaires ou d'extraordinaires. Les décisions ordinaires, telles que l'approbation des comptes, la fixation des dividendes et l'affectation des réserves, sont adoptées à la majorité simple. Les décisions extraordinaires, incluant la modification des statuts ou l'agrément de nouveaux associés, nécessitent la majorité absolue des associés et la majorité des trois quarts du capital social. Chaque exercice social dure douze mois, du 1er janvier au 31 décembre, le premier exercice prenant effet à la date d'immatriculation et se terminant le 31 décembre 2025. Sur les bénéfices annuels, une réserve légale obligatoire est constituée, et le surplus peut être réparti entre les associés ou attribué au gérant selon décision ordinaire. La société est dissoute à l'arrivée du terme ou par décision anticipée des associés représentant la majorité absolue et trois quarts du capital social. Fait à Tunis, en autant d'exemplaires que de droit."
    }
    
    
    # Chemin du PDF à remplir
    pdf_path = "formulaire_legal.pdf"  # Remplacer par votre fichier
    output_path = "formulaire_rempli.pdf"
    
    # Vérifier que le fichier existe
    if not Path(pdf_path).exists():
        print(f"❌ Fichier non trouvé: {pdf_path}")
        print("📝 Créez un fichier PDF ou modifiez le chemin dans le code")
    else:
        # Traiter le PDF
        result = process_pdf_with_ai(pdf_path, user_data, output_path)
        
        print("\n" + "="*50)
        print("📊 RÉSUMÉ")
        print("="*50)
        print(f"Succès: {result['success']}")
        print(f"Champs remplis: {len(result['filled_fields'])}")
        print(f"Champs manquants: {len(result['missing_fields'])}")
        
        if result['missing_fields']:
            print("\n⚠️ Champs non remplis:")
            for field in result['missing_fields']:
                print(f"  - {field}")