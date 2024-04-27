class DatasetRegistry:
    registry = {}

    @classmethod
    def register(cls, name):
        def inner_wrapper(wrapped_class):
            if name in cls.registry:
                raise ValueError(f"Dataset '{name}' already registered.")
            cls.registry[name] = wrapped_class
            print(f"Registered dataset {name}")  # Debugging statement
            return wrapped_class
        return inner_wrapper

    @classmethod
    def get_dataset(cls, config_dict, basedir, sequence, **kwargs):
        dataset_name = config_dict["dataset_name"].lower()
        if dataset_name in cls.registry:
            return cls.registry[dataset_name](config_dict, basedir, sequence, **kwargs)
        else:
            raise ValueError(f"Unknown dataset name {dataset_name}")