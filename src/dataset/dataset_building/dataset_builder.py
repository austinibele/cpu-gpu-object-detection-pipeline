from src.dataset.dataset_building.military_aircraft.military_aircraft_dataset_builder import MilitaryAircraftDatasetBuilder


class DatasetBuilder:
    @classmethod
    def build_dataset(cls, dataset_type):
        if dataset_type == "military_aircraft":
            builder = MilitaryAircraftDatasetBuilder()
            return builder.build_dataset()

    @classmethod
    def build_train_dataset(cls, dataset_type):
        pass
    
    @classmethod
    def build_val_dataset(cls, dataset_type):
        pass

