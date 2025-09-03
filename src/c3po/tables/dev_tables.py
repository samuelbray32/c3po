import datajoint as dj

from ..analysis.analysis import C3poAnalysis

schema = dj.schema("sb_c3po", locals())


@schema
class C3POStorage(dj.Manual):
    definition = """
    model_name: varchar(64)
    ---
    encoder_args: longblob
    context_args: longblob
    rate_args: longblob
    latent_dim: int
    context_dim: int
    input_shape: longblob
    learned_params: longblob
    """

    def fetch_analysis_object(self):
        return C3poAnalysis(init_key=self.fetch1("KEY"))
