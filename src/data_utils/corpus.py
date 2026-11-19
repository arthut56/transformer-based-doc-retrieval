import os
import shutil
from datasets import Dataset, DatasetDict, load_from_disk
from src.data_utils.download_retrieval_datasets import get_fiqa_dataset, get_squad_dataset

class Corpus:
    def __init__(self, dataset_name: str, base_dir: str):
        self.dataset_name = dataset_name
        if not os.path.exists(f'{base_dir}/{self.dataset_name}'):
            raw_data_dir = f'{base_dir}/raw'
            os.makedirs(raw_data_dir, exist_ok=True)
            # load and process data
            self.ds = self.__load_data(base_dir=raw_data_dir)
            self.__rename_columns()
            self.relevant_ids = self.__assign_ids()
            os.makedirs(f'{base_dir}/{self.dataset_name}', exist_ok=True)
            DatasetDict(self.ds).save_to_disk(f'{base_dir}/{self.dataset_name}')
            print(f'Saved {self.dataset_name} dataset to path {base_dir}/{self.dataset_name}.')
            shutil.rmtree(f'{base_dir}/raw')
        else:
            self.ds = load_from_disk(f'{base_dir}/{self.dataset_name}')
            self.relevant_ids = {qid: doc_ids
                                 for split in self.ds.values()
                                 for qid, doc_ids in zip(split['query_id'], split['document_id'])}


    @property
    def train(self):
        return self.ds.get('train', None)
    
    @property
    def val(self):
        return self.ds.get('validation', None)

    @property
    def test(self):
        return self.ds.get('test', None)

    @property
    def all_documents(self):
        """
        Every unique document across all splits. A document relevant to several queries
        (e.g. a SQuAD context shared by ~5 questions) appears once.
        """
        all_text, all_ids, seen = [], [], set()
        for split in (self.train, self.val, self.test):
            if split is None:
                continue
            for texts, ids in zip(split['text'], split['document_id']):
                for text, doc_id in zip(texts, ids):
                    if doc_id not in seen:
                        seen.add(doc_id)
                        all_text.append(text)
                        all_ids.append(doc_id)

        # One document per row, each as a 1-element list (the row format DocumentDataset reads)
        return Dataset.from_dict({'text': [[text] for text in all_text],
                                  'document_id': [[doc_id] for doc_id in all_ids]})


    def __load_data(self, base_dir) -> dict[Dataset]:
        assert self.dataset_name in ['squad', 'fiqa'], f'Dataset {self.dataset_name} not supported. Options are ["squad", "fiqa"]'
        
        if self.dataset_name == 'fiqa':
            ds = get_fiqa_dataset(base_dir)

        elif self.dataset_name == 'squad':
            ds = get_squad_dataset(base_dir)

        return {key: ds[key] for key in ds.keys()}

    def __rename_columns(self) -> dict[Dataset]:
        column_mapping = {'fiqa':  'ground_truths',
                          'squad': 'context'
                         }
        
        for key in self.ds:
            if self.dataset_name == 'squad':
                # SQuAD has one context string per question; wrap it in a list so 'text'
                # always holds a list of relevant documents, as in FiQA.
                self.ds[key] = self.ds[key].map(lambda row: {'text': [row['context']]},
                                                remove_columns=['context'])
            else:
                self.ds[key] = self.ds[key].rename_column(column_mapping[self.dataset_name], 'text')
        return self.ds

    def __assign_ids(self):
        """
        Give each query a unique id and each unique document text one id. Identical texts
        share an id, so a SQuAD context asked about by several questions is a single document
        and retrieving it counts as a hit for all of them.
        """
        q_id_counter = 0
        doc_text_to_id = {}
        relevant_document_ids = {}

        for key in self.ds:
            query_ids = []
            document_ids = []
            for documents in self.ds[key]['text']:
                query_ids.append(f'q{q_id_counter}')
                relevant_ids = []
                for doc in documents:
                    if doc not in doc_text_to_id:
                        doc_text_to_id[doc] = f'doc{len(doc_text_to_id)}'
                    relevant_ids.append(doc_text_to_id[doc])

                document_ids.append(relevant_ids)

                relevant_document_ids[f'q{q_id_counter}'] = relevant_ids
                q_id_counter += 1

            self.ds[key] = self.ds[key].add_column('query_id', query_ids)
            self.ds[key] = self.ds[key].add_column('document_id', document_ids)

        return relevant_document_ids
