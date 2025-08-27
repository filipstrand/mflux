from mflux.models.flux.variants.dreambooth.dataset.batch import Example


class DreamBoothPreProcessing:
    @staticmethod
    def augment(example: Example) -> list[Example]:
        # Currently this does nothing.
        return [example]
