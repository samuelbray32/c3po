import datajoint as dj
from .model.encoder import encoder_factory
from .model.context import context_factory
from .model.model import Embedding as _Embedding
from .model.model import C3PO as _C3PO
from .model.rate_prediction import rate_prediction_factory
from .model.model import loss
from .model.util import MLP

schema = dj.schema("c3po", locals())


@schema
class EncoderParams(dj.Manual):
    definition = """
    encoder_model: str
    encoder_params_name: str
    ---
    encoder_params: longblob # dictionary of encoder parameters
    """


@schema
class Encoder(dj.Manual):
    definition = """
    -> EncoderParams
    latent_dim: int
    """

    def create_encoder(self, key):
        key = (EncoderParams & (self & key)).fetch1()
        latent_dim = (self & key).fetch1("latent_dim")
        return encoder_factory(
            encoder_model=key["encoder_model"],
            latent_dim=latent_dim,
            **key["encoder_params"]
        )


@schema
class ContextParams(dj.Manual):
    definition = """
    context_model: str
    context_params_name: str
    ---
    context_params: longblob
    """


@schema
class Context(dj.Manual):
    definition = """
    -> ContextParams
    context_dim: int
    """

    def create_context(self, key):
        key = (ContextParams & (self & key)).fetch1()
        context_dim = (self & key).fetch1("context_dim")
        return context_factory(
            context_model=key["context_model"],
            context_dim=context_dim,
            **key["context_params"]
        )


@schema
class Embedding(dj.Manual):
    definition = """
    -> Encoder
    -> Context
    """

    def create_embedding(self, key):
        key = (self & key).fetch1("KEY")
        encoder_model, encoder_params = (EncoderParams & key).fetch1(
            "encoder_model", "encoder_params"
        )
        latent_dim = (Encoder & key).fetch1("latent_dim")
        context_model, context_params = (ContextParams & key).fetch1(
            "context_model", "context_params"
        )
        context_dim = (Context & key).fetch1("context_dim")
        return _Embedding(
            latent_dim=latent_dim,
            context_dim=context_dim,
            encoder_args={"encoder_model": encoder_model, **encoder_params},
            context_args={"context_model": context_model, **context_params},
        )


@schema
class RatePredictionParams(dj.Manual):
    definition = """
    rate_prediction_model: str
    rate_prediction_params_name: str
    ---
    rate_prediction_params: longblob
    """


@schema
class RatePrediction(dj.Manual):
    definition = """
    -> RatePredictionParams
    latent_dim: int
    context_dim: int
    """

    def create_rate_prediction(self, key):
        key = (RatePredictionParams & (self & key)).fetch1()
        latent_dim = (self & key).fetch1("latent_dim")
        context_dim = (self & key).fetch1("context_dim")
        return rate_prediction_factory(
            rate_model=key["rate_prediction_model"],
            latent_dim=latent_dim,
            context_dim=context_dim,
            **key["rate_prediction_params"]
        )


@schema
class C3PO(dj.Manual):
    definition = """
    -> Embedding
    -> RatePrediction
    """

    def create_c3po(self, key, n_neg_samples=10):
        # Note: n_neg samples is a training hyperparameter, rather than a model hyperparameter
        key = (self & key).fetch1("KEY")
        encoder_args = {
            **(EncoderParams & key).fetch1("encoder_params"),
            "encoder_model": key["encoder_model"],
        }
        context_args = {
            **(ContextParams & key).fetch1("context_params"),
            "context_model": key["context_model"],
        }
        rate_args = {
            **(RatePredictionParams & key).fetch1("rate_prediction_params"),
            "rate_model": key["rate_prediction_model"],
        }
        return _C3PO(
            encoder_args=encoder_args,
            context_args=context_args,
            rate_args=rate_args,
            latent_dim=key["latent_dim"],
            context_dim=key["context_dim"],
            n_neg_samples=n_neg_samples,
        )


@schema
class TrainingDataset(dj.Manual):
    definition = """
    dataset_name: str
    ---
    dataset: longblob
    """


@schema
class TrainingParams(dj.Manual):
    definition = """
    training_params_name: str
    ---
    n_neg_samples: int
    training_params: longblob
    """


@schema
class C3POTrainingSelection(dj.Manual):
    definition = """
    -> C3PO
    -> TrainingDataset
    -> TrainingParams
    """


@schema
class C3POTraining(dj.Manual):
    definition = """
    -> C3POTrainingSelection
    ---
    training_loss: float
    fit_params: longblob
    """

    def make(self, key):
        c3po = (C3PO & key).create_c3po()
        dataset = (TrainingDataset & key).load_data()
        # TODO: Implement training
        # initialize params
        # run training loop
        # save trained params

    def embed(self, x, delta_t):
        key = self.fetch1("KEY")
        embedding = (Embedding & key).create_embedding()
        params = (self).fetch1("fit_params")
        embedding_params = False  # TODO: parse params
        return embedding.apply(x, delta_t, embedding_params)
