# Détection et Reconnaissance de pièces d'euros

Ce projet implémente un système de détection et de reconnaissance de pièces en euros en utilisant des techniques de vision par ordinateur et d'apprentissage automatique. 

Ce projet a été réalisé dans le cadre de l'UE **Image** de M. *LOBRY Sylvain*, au sein de l'*Université Paris Cité*.
## Installation

1. Installez les dépendances :
    ```sh
    pip install -r requirements.txt
    ```

## Utilisation

0. Impossible de déposer sur Moodle le modele, ainsi que les images (trop lourd), voici un lien Google Drive contenant ces fichiers si besoin : 

https://drive.google.com/drive/folders/1Ygf2MWsO2qJ043zj4rWwSY6QDpGg3C5y?usp=sharing

Le dossier Images est à mettre au même niveau que les dossiers scripts, resources, README etc..

1. Pour exécuter le script principal, naviguez dans le répertoire `ProjetImage/ProjetImage/scripts` :
    ```sh
    cd ProjetImage/ProjetImage/scripts
    python main.py
    ```

2. Vous serez invité à choisir une option. Voici les options disponibles :
    ```plaintext
    Insert option to continue:
          1. Split datasets
          2. Extract labels from dataset
          3. Test model
          4. Generate squared images from dataset
          5. Test detect circles
          6. (Training) Find Hough Parameters
          7. Run coin detection on one image
          8. Run coin recognition on one image
          9. Test coin detection on testing set
          10. Test coin recognition on testing set
    ```

   Choisissez l'option en entrant le numéro correspondant.


# Auteurs

FEKIH HASSEN Yassine
CALEGARI Murilo

