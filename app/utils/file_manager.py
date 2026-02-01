import os

class FileManager:
    """
    Olyan osztály, amely fájlok kezelésére szolgál: létrehozás, írás, olvasás, törlés stb.
    """

    def __init__(self, path):
        """
        Inicializálja az osztályt egy fájl elérési útjával.

        :param path: A fájl elérési útvonala
        """
        self.path = path

    def create_file(self, content=""):
        """
        Létrehozza a fájlt, és ha van megadott tartalom, beleírja.

        :param content: A fájlba írandó szöveg (alapértelmezett: üres)
        """
        with open(self.path, "w", encoding="utf-8") as f:
            f.write(content)

    def read_file(self):
        """
        Beolvassa a fájl tartalmát.

        :return: A fájl tartalma szövegként
        """
        if os.path.isfile(self.path):
            with open(self.path, "r", encoding="utf-8") as f:
                return f.read()
        else:
            print("A fájl nem létezik.")
            return None


    def delete_file(self, file_path) -> None:
        """
        Törli a fájlt.
        """
        if os.path.isfile(file_path):
            os.remove(file_path)
            msg = f"{file_path} has been deleted."
            self.log(msg)
        else:
            msg = f"{file_path} does not exist."
            self.log(msg)

    def append_to_file(self, content):
        """
        Hozzáfűz szöveget a fájl végéhez.

        :param content: A hozzáfűzendő szöveg
        """
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(content)

    def file_exists(self):
        """
        Ellenőrzi, hogy a fájl létezik-e.

        :return: True ha létezik, különben False
        """
        return os.path.isfile(self.path)
