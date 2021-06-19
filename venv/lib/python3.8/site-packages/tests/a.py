from lexrankr import LexRank

from tokenizers import OktTokenizer


tokenizer = OktTokenizer()
# lexrank = LexRank(tokenizer, clustering_method='birch')
# lexrank = LexRank(tokenizer, clustering_method='dbscan')
lexrank = LexRank(tokenizer, clustering_method='affinity_propagation')
text = "사과 배 감 귤. 배 감 귤 수박. 감 귤 수박 딸기. 오이 참외 오징어. 참외 오징어 달팽이. 빨강 파랑 초록. 파랑 초록 노랑. 노랑 노랑 빨강. 검정 파랑 빨강 초록."

lexrank.summarize(text, no_below=0)
summaries = lexrank.probe()

# for cluster in lexrank.clusters:
# print("-", cluster)
# print()

# print(1 - lexrank.matrix)
# print(lexrank.matrix)
# print()

print(summaries)
# print()

# print(lexrank.index2cids)
# print(lexrank.cid2sentences)
# print(lexrank._clustering_model)
# print(lexrank.clustering_method)
