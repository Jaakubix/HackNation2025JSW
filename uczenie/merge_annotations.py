"""
Skrypt do łączenia adnotacji z dwóch źródeł i naprawy ścieżek plików.
"""
import json
from pathlib import Path
import os

def extract_filename(file_path):
    """Wyciąga nazwę pliku z pełnej ścieżki (obsługuje Windows i Linux)."""
    # Obsługa różnych formatów ścieżek
    path = file_path.replace("\\", "/")
    return Path(path).name

def merge_annotations(anno1_path, anno2_path, output_path, images_dir):
    """
    Łączy dwa pliki adnotacji COCO i naprawia ścieżki do obrazów.
    """
    # Wczytaj oba pliki
    with open(anno1_path, 'r', encoding='utf-8') as f:
        data1 = json.load(f)
    
    with open(anno2_path, 'r', encoding='utf-8') as f:
        data2 = json.load(f)
    
    # Zbierz dostępne pliki w katalogu images
    available_files = set(os.listdir(images_dir))
    print(f"Dostępne obrazy w {images_dir}: {len(available_files)}")
    
    # Ujednolicenie kategorii - użyj kategorii z większego zbioru
    # Sprawdź które kategorie są wspólne
    categories_map = {}
    all_categories = []
    
    # Zbierz wszystkie kategorie z obu źródeł
    for cat in data1.get('categories', []):
        categories_map[cat['name']] = cat['id']
        all_categories.append(cat)
    
    for cat in data2.get('categories', []):
        if cat['name'] not in categories_map:
            new_id = max(categories_map.values()) + 1 if categories_map else 0
            categories_map[cat['name']] = new_id
            all_categories.append({'id': new_id, 'name': cat['name']})
    
    print(f"Kategorie: {categories_map}")
    
    # Przetwórz obrazy i adnotacje z pierwszego źródła
    merged_images = []
    merged_annotations = []
    image_id_counter = 0
    anno_id_counter = 0
    
    # Mapowanie starych ID obrazów na nowe
    old_to_new_image_id = {}
    
    for img in data1.get('images', []):
        filename = extract_filename(img['file_name'])
        if filename in available_files:
            old_id = img['id']
            new_img = {
                'id': image_id_counter,
                'file_name': filename,
                'width': img['width'],
                'height': img['height']
            }
            merged_images.append(new_img)
            old_to_new_image_id[('data1', old_id)] = image_id_counter
            image_id_counter += 1
        else:
            print(f"Brak obrazu z data1: {filename}")
    
    # Adnotacje z pierwszego źródła
    for anno in data1.get('annotations', []):
        old_img_id = anno['image_id']
        key = ('data1', old_img_id)
        if key in old_to_new_image_id:
            new_anno = anno.copy()
            new_anno['id'] = anno_id_counter
            new_anno['image_id'] = old_to_new_image_id[key]
            merged_annotations.append(new_anno)
            anno_id_counter += 1
    
    # Przetwórz obrazy i adnotacje z drugiego źródła
    for img in data2.get('images', []):
        filename = extract_filename(img['file_name'])
        if filename in available_files:
            old_id = img['id']
            new_img = {
                'id': image_id_counter,
                'file_name': filename,
                'width': img['width'],
                'height': img['height']
            }
            merged_images.append(new_img)
            old_to_new_image_id[('data2', old_id)] = image_id_counter
            image_id_counter += 1
        else:
            print(f"Brak obrazu z data2: {filename}")
    
    # Adnotacje z drugiego źródła
    for anno in data2.get('annotations', []):
        old_img_id = anno['image_id']
        key = ('data2', old_img_id)
        if key in old_to_new_image_id:
            new_anno = anno.copy()
            new_anno['id'] = anno_id_counter
            new_anno['image_id'] = old_to_new_image_id[key]
            merged_annotations.append(new_anno)
            anno_id_counter += 1
    
    # Utwórz wynikowy słownik
    merged_data = {
        'images': merged_images,
        'categories': all_categories,
        'annotations': merged_annotations
    }
    
    # Zapisz wynik
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n=== PODSUMOWANIE ===")
    print(f"Obrazy: {len(merged_images)}")
    print(f"Adnotacje: {len(merged_annotations)}")
    print(f"Kategorie: {len(all_categories)}")
    print(f"Zapisano do: {output_path}")
    
    return merged_data

if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).parent.parent
    
    # Ścieżki do plików
    anno1 = PROJECT_ROOT / "annotations" / "result.json"  # Twoje wcześniejsze adnotacje
    anno2 = PROJECT_ROOT / "data" / "result.json"         # Nowe adnotacje z pendrive
    output = PROJECT_ROOT / "data" / "dataset_combined" / "annotations.json"
    images_dir = PROJECT_ROOT / "data" / "dataset_combined" / "images"
    
    merge_annotations(anno1, anno2, output, images_dir)
