import torch.nn as nn
import torch
import torch.nn.functional as F
from multimodal_transformers.model.tabular_config import TabularConfig
from multimodal_transformers.model.tabular_combiner import TabularFeatCombiner
from multimodal_transformers.model.layer_utils import MLP, calc_mlp_dims, hf_loss_func
from dataclasses import dataclass, field
from x_transformers import TransformerWrapper, Encoder



@dataclass
class MultimodalDataTrainingArguments:
  """
  Arguments pertaining to how we combine tabular features
  Using `HfArgumentParser` we can turn this class
  into argparse arguments to be able to specify them on
  the command line.
  """

  data_path: str = field(metadata={'help': 'the path to the csv file containing the dataset'})
  column_info_path: str = field(
      default=None,
      metadata={
          'help': 'the path to the json file detailing which columns are text, categorical, numerical, and the label'
  })

  column_info: dict = field(
      default=None,
      metadata={
          'help': 'a dict referencing the text, categorical, numerical, and label columns'
                  'its keys are text_cols, num_cols, cat_cols, and label_col'
  })

  categorical_encode_type: str = field(default='ohe',
                                        metadata={
                                            'help': 'sklearn encoder to use for categorical data',
                                            'choices': ['ohe', 'binary', 'label', 'none']
                                        })
  numerical_transformer_method: str = field(default='yeo_johnson',
                                            metadata={
                                                'help': 'sklearn numerical transformer to preprocess numerical data',
                                                'choices': ['yeo_johnson', 'box_cox', 'quantile_normal', 'none']
                                            })
  task: str = field(default="classification",
                    metadata={
                        "help": "The downstream training task",
                        "choices": ["classification", "regression"]
                    })

  mlp_division: int = field(default=4,
                            metadata={
                                'help': 'the ratio of the number of '
                                        'hidden dims in a current layer to the next MLP layer'
                            })
  combine_feat_method: str = field(default='individual_mlps_on_cat_and_numerical_feats_then_concat',
                                    metadata={
                                        'help': 'method to combine categorical and numerical features, '
                                                'see README for all the method'
                                    })
  mlp_dropout: float = field(default=0.1,
                              metadata={
                                'help': 'dropout ratio used for MLP layers'
                              })
  numerical_bn: bool = field(default=True,
                              metadata={
                                  'help': 'whether to use batchnorm on numerical features'
                              })
  use_simple_classifier: str = field(default=True,
                                      metadata={
                                          'help': 'whether to use single layer or MLP as final classifier'
                                      })
  mlp_act: str = field(default='relu',
                        metadata={
                            'help': 'the activation function to use for finetuning layers',
                            'choices': ['relu', 'prelu', 'sigmoid', 'tanh', 'linear']
                        })
  gating_beta: float = field(default=0.2,
                              metadata={
                                  'help': "the beta hyperparameters used for gating tabular data "
                                          "see https://www.aclweb.org/anthology/2020.acl-main.214.pdf"
                              })

  def __post_init__(self):
      assert self.column_info != self.column_info_path
      if self.column_info is None and self.column_info_path:
          with open(self.column_info_path, 'r') as f:
              self.column_info = json.load(f)

class DeepJIT(nn.Module):
    def __init__(self, args):
        super(DeepJIT, self).__init__()
        self.args = args

        V_msg = args.vocab_msg
        V_code = args.vocab_code
        Dim = args.embedding_dim
        Class = args.class_num        

        Ci = 1  # input of convolutional layer
        Co = args.num_filters  # output of convolutional layer
        Ks = args.filter_sizes  # kernel sizes

        # CNN-2D for commit message
        self.embed_msg = nn.Embedding(V_msg, Dim)
        self.convs_msg = nn.ModuleList([nn.Conv2d(Ci, Co, (K, Dim)) for K in Ks])

        # CNN-2D for commit code
        self.embed_code = nn.Embedding(V_code, Dim)
        self.convs_code_line = nn.ModuleList([nn.Conv2d(Ci, Co, (K, Dim)) for K in Ks])
        self.convs_code_file = nn.ModuleList([nn.Conv2d(Ci, Co, (K, Co * len(Ks))) for K in Ks])

        # other information
        self.dropout = nn.Dropout(args.dropout_keep_prob)
        self.fc1 = nn.Linear(2 * len(Ks) * Co, args.hidden_units)  # hidden units
        self.fc2 = nn.Linear(args.hidden_units, Class)
        self.sigmoid = nn.Sigmoid()

        self.num_labels = 1
        text_cols = ['code']
        cat_cols = ['ns', 'nf', 'fix', 'nd']
        numerical_cols = ['entrophy', 'age', 'rexp', 'la', 'ld', 'lt', 'nuc', 'exp', 'sexp']

        column_info_dict = {
            'text_cols': text_cols,
            'num_cols': numerical_cols,
            'cat_cols': cat_cols,
            'label_col': 'bug',
            'label_list': [0, 1]
        }
        # combiner_mode = 'gating_on_cat_and_num_feats_then_sum'
        combiner_mode = args.combiner
        data_args = MultimodalDataTrainingArguments(
            data_path='.',
            combine_feat_method=combiner_mode,
            column_info=column_info_dict,
            task='classification'
        )
        self.tabular_config = TabularConfig(num_labels=self.num_labels,
                                       cat_feat_dim=args.category_feat_dim,
                                       numerical_feat_dim=args.num_feat_dim,
                                       **vars(data_args))
        self.tabular_config.text_feat_dim = 384
        self.tabular_config.hidden_dropout_prob = args.dropout_keep_prob

        self.tabular_combiner = TabularFeatCombiner(self.tabular_config)

        ######
        # combiner_mode2 = 'gating_on_cat_and_num_feats_then_sum'
        # data_args2 = MultimodalDataTrainingArguments(
        #     data_path='.',
        #     combine_feat_method=combiner_mode2,
        #     column_info=column_info_dict,
        #     task='classification'
        # )
        # self.tabular_config2 = TabularConfig(num_labels=self.num_labels,
        #                                     cat_feat_dim=args.category_feat_dim,
        #                                     numerical_feat_dim=args.num_feat_dim,
        #                                     **vars(data_args2))
        # self.tabular_config2.text_feat_dim = 384
        # self.tabular_config2.hidden_dropout_prob = args.dropout_keep_prob
        # self.tabular_combiner2 = TabularFeatCombiner(self.tabular_config2)

        ###
        # self.tabular_config.use_simple_classifier = True
        # if self.tabular_config.use_simple_classifier:
        #     self.tabular_classifier = nn.Linear(384,
        #                                         self.tabular_config.num_labels)
        # else:
        #     dims = calc_mlp_dims(384,
        #                          division=self.tabular_config.mlp_division,
        #                          output_dim=self.tabular_config.num_labels)
        #     self.tabular_classifier = MLP(384,
        #                                   self.tabular_config.num_labels,
        #                                   num_hidden_lyr=len(dims),
        #                                   dropout_prob=self.tabular_config.mlp_dropout,
        #                                   hidden_channels=dims,
        #                                   bn=True)
        if args.combiner == 'concat':
            # self.fc1_2 = nn.Linear(467, args.hidden_units)
            # self.fc1_2 = nn.Linear(532, args.hidden_units)
            # self.fc1_2 = nn.Linear(495, args.hidden_units)
            # self.fc1_2 = nn.Linear(504, args.hidden_units)
            # self.fc1_2 = nn.Linear(531, args.hidden_units)
            self.fc1_2 = nn.Linear(542, args.hidden_units)


        elif args.combiner == 'individual_mlps_on_cat_and_numerical_feats_then_concat':
            self.fc1_2 = nn.Linear(425, args.hidden_units)
        elif args.combiner == 'mlp_on_concatenated_cat_and_numerical_feats_then_concat':
            self.fc1_2 = nn.Linear(393, args.hidden_units)  # hidden units
        elif args.combiner  == 'mlp_on_categorical_then_concat':
            self.fc1_2 = nn.Linear(430, args.hidden_units)  # hidden units
            # self.fc1_2 = nn.Linear(462, args.hidden_units)  # hidden units
            # self.fc1_2 = nn.Linear(444, args.hidden_units)  # hidden units
            # self.fc1_2 = nn.Linear(448, args.hidden_units)
            # self.fc1_2 = nn.Linear(462, args.hidden_units)
            # self.fc1_2 = nn.Linear(467, args.hidden_units)
        else:
            self.fc1_2 = nn.Linear(384, args.hidden_units)


        ##
        self.cat_transformer = TransformerWrapper(
            num_tokens=256,
            max_seq_len=150,
            attn_layers=Encoder(
                dim=128,
                depth=2,
                heads=4
            )
        )

        self.pooler = nn.AdaptiveAvgPool2d((args.category_feat_dim, 1))


    def forward_msg(self, x, convs):
        # note that we can use this function for commit code line to get the information of the line
        x = x.unsqueeze(1)  # (N, Ci, W, D)
        x = [F.relu(conv(x)).squeeze(3) for conv in convs]  # [(N, Co, W), ...]*len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)
        x = torch.cat(x, 1)
        return x

    def forward_code(self, x, convs_line, convs_hunks):
        n_batch, n_file = x.shape[0], x.shape[1]
        x = x.reshape(n_batch * n_file, x.shape[2], x.shape[3])

        # apply cnn 2d for each line in a commit code
        x = self.forward_msg(x=x, convs=convs_line)

        # apply cnn 2d for each file in a commit code
        x = x.reshape(n_batch, n_file, self.args.num_filters * len(self.args.filter_sizes))
        x = self.forward_msg(x=x, convs=convs_hunks)
        return x

    # def forward(self, msg, code):
    def forward(self, msg, code, cat_feats, numerical_feats, labels):
        x_msg = self.embed_msg(msg)
        x_msg = self.forward_msg(x_msg, self.convs_msg)

        x_code = self.embed_code(code)
        x_code = self.forward_code(x_code, self.convs_code_line, self.convs_code_file)

        x_commit = torch.cat((x_msg, x_code), 1)
        x_commit = self.dropout(x_commit)

        # x_commit = x_commit.to(torch.float32)


        #### Hand Features
        numerical_feats = numerical_feats.to(torch.float32)
        cat_feats = cat_feats.to(torch.float32)
        # cat_mask = torch.ones_like(cat_feats).bool()
        # cat_feats = self.cat_transformer(cat_feats.long(), mask=cat_mask)
        # cat_feats = self.pooler(cat_feats).squeeze(2)
        # cat_feats = self.dropout(cat_feats)

        combined_feats = self.tabular_combiner(x_commit, cat_feats, numerical_feats)
        # print("combined_feats:", combined_feats.size())
        # combined_feats2 = self.tabular_combiner2(x_commit, cat_feats, numerical_feats)
        #
        # loss, logits, classifier_layer_outputs = hf_loss_func(combined_feats,
        #                                                       self.tabular_classifier,
        #                                                       labels,
        #                                                       self.num_labels,
        #                                                       class_weights=None)
        #
        # logits = self.sigmoid(logits).squeeze(1)
        # return logits

        combined_feats = self.dropout(combined_feats)
        out = self.fc1_2(combined_feats)
        out = F.relu(out)
        # out = self.dropout(out)
        out = self.fc2(out)
        out = self.sigmoid(out).squeeze(1)
        return out



        # #### No Hand Features
        # out = self.fc1(x_commit)
        # out = F.relu(out)
        # out = self.fc2(out)
        # out = self.sigmoid(out).squeeze(1)
        # return out