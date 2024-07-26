import subprocess

import cv2
import os
import numpy as np
from pydub import AudioSegment
from pydub.utils import mediainfo


class FileManager:
    def __init__(self, video_path, output_dir="out", audio_path="audio", output_vid="out.mp4"):
        self.video_path = video_path
        self.audio_path = audio_path
        self.output_dir = output_dir
        self.output_vid = output_vid

    def extract_audio(self):
        """
        Extrait l'audio d'une vidéo et le sauvegarde dans un fichier.

        Args:
            video_path (str): Chemin vers le fichier vidéo.
            audio_path (str): Chemin vers le fichier audio à sauvegarder.
        """
        audio = AudioSegment.from_file(self.video_path)
        audio.export(self.audio_path, format="wav")

    def analyze_audio(self):
        """
        Analyse l'audio pour obtenir le volume par segment temporel.

        Args:
            audio_path (str): Chemin vers le fichier audio.

        Returns:
            list: Liste des volumes pour chaque segment de 1 seconde.
        """
        audio = AudioSegment.from_file(self.audio_path)
        duration_ms = len(audio)
        segment_length_ms = 1000  # 1 seconde
        volumes = []

        for start_ms in range(0, duration_ms, segment_length_ms):
            segment = audio[start_ms:start_ms + segment_length_ms]
            volumes.append(segment.dBFS)

        return volumes

    @staticmethod
    def pixelate_image(image, block_size, greyscale=False):
        """
        Pixellise une image en utilisant un bloc de pixels fourni.

        Args:
            image (numpy.ndarray): Image a pixelliser.
            block_size (int): Taille du bloc de pixels.
            greyscale (bool): Nuances de gris ou noir et blanc.

        Returns:
            numpy.ndarray: Image pixellisee.
        """
        height, width = image.shape[:2]
        # Redimensionner l'image pour pixelliser
        small = cv2.resize(image, (width // block_size, height // block_size), interpolation=cv2.INTER_LINEAR)
        if not greyscale:
            small = FileManager.soft_black_and_white(small)
        # Agrandir l'image pixellisée à sa taille originale
        pixelated = cv2.resize(small, (width, height), interpolation=cv2.INTER_NEAREST)
        return pixelated

    @staticmethod
    def blur_image(image, kernel_radius=0):
        """
        Applique un effet de flou gaussien a une image

        Args:
            image (numpy.ndarray): Image a flouter
            kernel_radius (int): Rayon du noyau >= 0

        Returns:
            numpy.ndarray: Image floutée.
        """
        # Vérifier que la taille du noyau est un nombre impair
        if not kernel_radius:
            return image

        kernel_size = kernel_radius * 2 + 1
        # Appliquer le flou gaussien
        blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        return blurred_image

    @staticmethod
    def soft_black_and_white(image):
        """
        Convertit une image en niveaux de gris en noir et blanc avec une distribution aléatoire

        Args:
            image (numpy.ndarray): Image à convertir.

        Returns:
            numpy.ndarray: Image en noir et blanc.
        """
        # Convertir l'image en niveaux de gris
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Normaliser l'image à une échelle de 0 à 1
        normalized_gray = gray_image / 255.0
        # Générer une distribution aléatoire
        random_matrix = np.random.random(normalized_gray.shape)
        # Créer une image binaire avec la distribution aléatoire
        black_and_white = np.where(normalized_gray > random_matrix, 255, 0).astype(np.uint8)

        # masques de seuils
        white_mask = gray_image >= 250.0
        black_mask = gray_image <= 5.0
        black_and_white[white_mask] = 255.0
        black_and_white[black_mask] = 0.0

        return black_and_white

    def extract_to_video(self, greyscale=True, block_size=0, blur_radius=0):
        """
        Extrait des frames d'une vidéo à une fréquence spécifiée et pixellise les images.

        Args:
            fps_arg (float): nombre d'images par seconde a extraire.
            greyscale (bool): Autorise les nuances de gris
            block_size (int): Taille des blocs pour la pixellisation.
        """

        # Ouvrir la vidéo
        cap = cv2.VideoCapture(self.video_path)

        if not cap.isOpened():
            print(f"Erreur lors de l'ouverture de la vidéo: {self.video_path}")
            return

        # Récupérer les propriétés de la vidéo
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"FPS de la vidéo: {fps}")
        print(f"Nombre total de frames: {total_frames}")
        print("Extraction des images")

        # Créer le dossier de sortie si il n'existe pas
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        frame_number = 0
        checkpoint = total_frames // 100
        processed_frames = []

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            # Extraire et sauvegarder les frames
            if blur_radius > 0:
                frame = FileManager.blur_image(frame, blur_radius)
            if block_size > 1:
                frame = FileManager.pixelate_image(frame, block_size)
            if greyscale:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                frame = FileManager.soft_black_and_white(frame)
            processed_frames.append(frame)
            frame_number += 1
            if frame_number % checkpoint == 0:
                print('.', end='')

        cap.release()
        print("\nCreation de la video")

        # Recréer la vidéo à partir des frames transformées
        output_video_path = "./temp.mp4"
        height, width = processed_frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height), False)  # False for grayscale

        frame_number = 0
        for frame in processed_frames:
            video_writer.write(frame)
            frame_number += 1
            if frame_number % checkpoint == 0:
                print('.', end='')

        video_writer.release()
        print("\nVideo creee, ajout du son...")

        # Ajouter l'audio d'origine à la nouvelle vidéo
        final_output_path = self.output_vid
        command = [
            'ffmpeg', '-y', '-i', output_video_path, '-i', self.video_path,
            '-c:v', 'copy', '-c:a', 'aac', '-map', '0:v:0', '-map', '1:a:0', '-vcodec', 'libx264', '-crf', '1',
            final_output_path
        ]
        subprocess.run(command, check=True)
        print(f"Video sauvegardee sous {final_output_path} !")
