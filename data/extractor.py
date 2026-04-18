import os
import shutil

src_dir = "data/PlantVillage"
dst_dir = "data/dataset_tomato"

CLASSES = [
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato_healthy"
]

def extract_classes(src_dir, dst_dir):
    print("📂 Source :", src_dir)
    print("📁 Destination :", dst_dir)

    if not os.path.exists(src_dir):
        print("❌ ERREUR : dossier source introuvable !")
        return

    os.makedirs(dst_dir, exist_ok=True)

    total_copied = 0

    for cls in CLASSES:
        src = os.path.join(src_dir, cls)
        dst = os.path.join(dst_dir, cls)

        if not os.path.exists(src):
            print(f"⚠️ Classe introuvable : {cls}")
            continue

        shutil.copytree(src, dst, dirs_exist_ok=True)

        nb_files = len(os.listdir(src))
        total_copied += nb_files

        print(f"✅ {cls} → {nb_files} images copiées")

    print(f"📊 Total images copiées : {total_copied}")

if __name__ == "__main__":
    print("🚀 Lancement du script extractor")
    extract_classes(src_dir, dst_dir)
    print("🎯 Extraction terminée avec succès")