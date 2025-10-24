import os
import os.path as osp
import pandas as pd
import torch
import numpy as np
from data.preprocessing import PreprocessingMixin
from torch_geometric.data import HeteroData
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import download_url
from torch_geometric.data import extract_zip
from torch_geometric.io import fs
from typing import Callable, List, Optional


class NewsAPI(InMemoryDataset):
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        super().__init__(root, transform, pre_transform,
                         force_reload=force_reload)
        self.load(self.processed_paths[0], data_cls=HeteroData)
    
    @property
    def raw_file_names(self) -> List[str]:
        return ['users.csv', 'articles.csv', 'click_events.csv']
    
    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    @property
    def has_process(self) -> bool:
        return not os.path.exists(self.processed_paths[0])
    
    def download(self) -> None:
        # Check if raw files already exist in the root directory
        raw_files = ['users.csv', 'articles.csv', 'click_events.csv']
        root_files = [f for f in raw_files if osp.exists(osp.join(self.root, f))]
        
        if root_files:
            # Move files from root to raw directory
            os.makedirs(self.raw_dir, exist_ok=True)
            for file in root_files:
                src = osp.join(self.root, file)
                dst = osp.join(self.raw_dir, file)
                if not osp.exists(dst):
                    os.rename(src, dst)
        else:
            # Files should already be in raw directory or need to be downloaded
            # For now, just ensure raw directory exists
            os.makedirs(self.raw_dir, exist_ok=True)

    def process():
        pass

CLICK_EVENTS_FIELDS = [
    "date",
    "search_uuid",
    "query",
    "position",
    "age",
    "is_click",
    "state_code",
    "device",
    "timestamp",
    "article_id",
    "user_id",
]

class RawNewsAPI(NewsAPI, PreprocessingMixin):
    def __init__(
        self,
        root,
        transform=None,
        pre_transform=None,
        force_reload=False,
        split=None
    ) -> None:
        super(RawNewsAPI, self).__init__(
            root, transform, pre_transform, force_reload
        )

    def _load_feedback_data(self):
        return pd.read_csv(self.raw_paths[2])[CLICK_EVENTS_FIELDS]
    
    def process(self, max_seq_len=None) -> None:
        print(self.raw_paths)
        data = HeteroData()
        click_events_df = self._load_feedback_data()
        print(click_events_df.columns)

        # Process article data:
        df = pd.read_csv(self.raw_paths[1])# , index_col='article_id')
        df = df.head(1000)
        print(click_events_df.shape)
        click_events_df = click_events_df.join(df, on='article_id', how='inner', rsuffix='_df')
        click_events_df = click_events_df.drop(columns=['article_id_df', 'title', 'category', 'total_articles'])
        print(f"updated click_events_df shape: {click_events_df.shape}")
        # df = df.set_index('article_id')
        
        # article_mapping = {idx: i for i, idx in enumerate(df.index)}
        # print(article_mapping)

        categories = self._process_category(df["category"].str.get_dummies('|').values, one_hot=False)
        categories = torch.from_numpy(categories).to(torch.float)

        titles_text = df["title"].apply(lambda s:  s.strip() if isinstance(s, str) else str(s).strip()).tolist()
        print(f"titles_text length: {len(titles_text)}")
        titles_emb = self._encode_text_feature(titles_text)
        print(f"titles_emb length: {titles_emb.shape}")

        x = torch.cat([titles_emb, categories], axis=1)

        data['item'].x = x
        # Process user data:
        # full_df = pd.DataFrame({"userId": click_events_df["user_id"].unique()})
        # df = self._remove_low_occurrence(click_events_df, full_df, "userId")
        user_mapping = {idx: i for i, idx in enumerate(click_events_df["user_id"])}
        print(f"user mapping length: {len(user_mapping)}")
        self.int_user_data = click_events_df

        # # Process rating data:
        # df = self._remove_low_occurrence(
        #     click_events_df,
        #     click_events_df,
        #     ["userId", "articleId"]
        # )
        src = [user_mapping[idx] for idx in click_events_df['user_id']]
        dst = [idx for idx in click_events_df['article_id']]
        edge_index = torch.tensor([src, dst])
        data['user', 'is_click', 'item'].edge_index = edge_index

        is_click = torch.from_numpy(click_events_df['is_click'].values).to(torch.long)
        data['user', 'is_click', 'item'].is_click = is_click

        time = torch.from_numpy(click_events_df['timestamp'].values)
        data['user', 'is_click', 'item'].time = time

        data['item', 'clicked_by', 'user'].edge_index = edge_index.flip([0])
        data['item', 'clicked_by', 'user'].is_click = is_click
        data['item', 'clicked_by', 'user'].time = time

        click_events_df["itemId"] = click_events_df.article_id

        click_events_df["is_click"] = (2*click_events_df["is_click"]).astype(int)
        data["user", "clicked", "item"].history = self._generate_user_history(
            click_events_df,
            features=["itemId", "is_click"],
            window_size=max_seq_len if max_seq_len is not None else 200,
            stride=180,
            train_split=0.8
        )
        data['item'].text = np.array(titles_text)
        print(f"data['item'].text length: {len(data['item'].text)}")
        print(data['item'].text[:10])
        print(f"titles_emb shape: {titles_emb.shape}")
        print(titles_emb[:10])

        gen = torch.Generator()
        gen.manual_seed(42)
        data['item'].is_train = torch.rand(titles_emb.shape[0], generator=gen) > 0.20
        filter = data['item'].is_train
        not_filter = ~data['item'].is_train
        print(f"is_train length: {len(data['item']['x'][filter])}")
        print(f"test length: {len(data['item']['x'][not_filter])}")
        print(data['item'].is_train[:10])

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        self.save([data], self.processed_paths[0])
