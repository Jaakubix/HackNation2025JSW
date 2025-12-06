from utils import upload_envs
upload_envs()

from train.step import Trainer

def runTraining():
    # Jeden krok: detekcja obiektów na bazie COCO
    trainer = Trainer()
    trainer.run()

def run():
    runTraining()

if __name__ == "__main__":
    run()
