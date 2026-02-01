import os
import shutil
from app.services.auxiliary import Auxiliary

class DirectoryManager:
    """
    Olyan osztály, amely különböző műveleteket végez mappákon.
    """

    def __init__(self):
        """
        Inicializálja az osztályt.
        """
        self.auxiliary = Auxiliary()

    def create_directory(self, directory_path):
        """
        Létrehozza a mappát, ha még nem létezik.
        """
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
        else:
            self.auxiliary.log("A mappa már létezik.")

    def delete_directory(self, directory_path):
        """
        Törli a mappát és annak teljes tartalmát.
        """
        if os.path.exists(directory_path) and os.path.isdir(directory_path):
            shutil.rmtree(directory_path)
        else:
            self.auxiliary.log(f"A {directory_path} mappa nem létezik.")

    def clear_directory(self, directory_path):
        """
        Törli a mappán belüli összes fájlt és almappát, de magát a mappát nem.
        """
        if os.path.exists(directory_path) and os.path.isdir(directory_path):
            for filename in os.listdir(directory_path):
                file_path = os.path.join(directory_path, filename)
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
        else:
            self.auxiliary.log(f"A {directory_path} mappa nem létezik.")

    @staticmethod
    def is_directory(directory_path):
        """
        Megvizsgálja, hogy az elérési út mappa-e.

        :return: True ha mappa, egyébként False
        """
        return os.path.isdir(directory_path)

    @staticmethod
    def is_empty(directory_path):
        """
        Ellenőrzi, hogy a mappa létezik-e és üres-e.

        :return: True ha üres, False ha nem vagy nem létezik
        """
        return os.path.isdir(directory_path) and not os.listdir(directory_path)
