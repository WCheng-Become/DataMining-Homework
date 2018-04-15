# -*- coding: utf-8 -*-
from itertools import chain, combinations
from collections import defaultdict

# 返回数组arr的非空子集
def subsets(arr):
    return chain(*[combinations(arr, i + 1) for i, a in enumerate(arr)])

# 返回满足最小支持度的子集
def returnItemsWithMinSupport(itemSet, transactionList, minSupport, freqSet):
        _itemSet = set()
        localSet = defaultdict(int)

        for item in itemSet:
                for transaction in transactionList:
                        if item.issubset(transaction):
                                freqSet[item] += 1
                                localSet[item] += 1

        for item, count in localSet.items():
                support = float(count)/len(transactionList)

                if support >= minSupport:
                        _itemSet.add(item)

        return _itemSet

# 返回指定长度的自连接集合
def joinSet(itemSet, length):
    return set([i.union(j) for i in itemSet for j in itemSet if len(i.union(j)) == length])

# 获取商品总集合 及 总交易记录
def getItemSetTransactionList(data_iterator):
    transactionList = list()
    itemSet = set()
    for record in data_iterator:
        transaction = frozenset(record)
        transactionList.append(transaction)
        for item in transaction:
            itemSet.add(frozenset([item]))              # Generate 1-itemSets
    return itemSet, transactionList

# 返回频繁项集:((频繁项集), 支持度)，关联规则:(((源频繁项集),(目标频繁项集)),支持度,置信度,lift)
def runApriori(data_iter, minSupport, minConfidence):
    itemSet, transactionList = getItemSetTransactionList(data_iter)
    # 所有项的频数 频繁项和非频繁项，1项和K项
    freqSet = defaultdict(int)
    # 存储各元频繁项集合 key=K，value=K项频繁项集合
    largeSet = dict()
    # Global dictionary which stores (key=n-itemSets,value=support)
    # which satisfy minSupport

    # assocRules = dict()
    # Dictionary which stores Association Rules
    # 一元频繁项集合
    oneCSet = returnItemsWithMinSupport(itemSet, transactionList, minSupport, freqSet)

    currentLSet = oneCSet
    k = 2
    # 递归逐层求解
    while(currentLSet != set([])):
        # print(k)
        largeSet[k-1] = currentLSet
        currentLSet = joinSet(currentLSet, k)
        currentCSet = returnItemsWithMinSupport(currentLSet, transactionList, minSupport, freqSet)
        currentLSet = currentCSet
        k = k + 1

    def getSupport(item):
            """local function which Returns the support of an item"""
            return float(freqSet[item])/len(transactionList)

    toRetItems = []
    for key, value in largeSet.items():
        toRetItems.extend([(tuple(item), getSupport(item))
                           for item in value])

    toRetRules = []
    for key, value in largeSet.items()[1:]:
        for item in value:
            _subsets = map(frozenset, [x for x in subsets(item)])
            for element in _subsets:
                remain = item.difference(element)
                if len(remain) > 0:
                    confidence = getSupport(item)/getSupport(element)
                    if confidence >= minConfidence:
                        toRetRules.append(((tuple(element), tuple(remain)),getSupport(item),
                                           confidence, confidence/getSupport(remain)))
    return toRetItems, toRetRules


def saveResults(items, rules):
    fre_items_file = open(r'frequent_items.txt', 'w')
    ass_rules_by_confidence = open(r'association_rules_by_confidence.txt', 'w')
    ass_rules_by_lift = open(r'association_rules_by_lift.txt', 'w')

    fre_items_file.write('frequent_item_set\tsupport\n')
    # 以support降序排列
    for item, support in sorted(items, key=lambda (item, support): support, reverse=True):
        fre_items_file.write("%s\t%.3f\n" % (str(item), support))
    fre_items_file.close()

    # 以confidence降序排列
    ass_rules_by_confidence.write('rule\tsupport\tconfidence\tlift\n')
    for rule, support, confidence, lift in sorted(rules, key=lambda (rule, suppport, confidence, lift): confidence, reverse=True):
        pre, post = rule
        ass_rules_by_confidence.write("%s ==> %s\t%.3f\t%.3f\t%.3f\n" % (str(pre), str(post), support, confidence, lift))
    ass_rules_by_confidence.close()

    # 以lift降序排列
    ass_rules_by_lift.write('rule\tsupport\tconfidence\tlift\n')
    for rule, support, confidence, lift in sorted(rules, key=lambda (rule, suppport, confidence, lift): lift, reverse=True):
        pre, post = rule
        ass_rules_by_lift.write("%s ==> %s\t%.3f\t%.3f\t%.3f\n" % (str(pre), str(post), support, confidence, lift))
    ass_rules_by_lift.close()

# 读取记录文件
def dataFromFile(fname):
        file_iter = open(fname, 'rU')
        for line in file_iter:
                line = line.strip().rstrip(',')
                shopping_basket = line.split(',')
                while '' in shopping_basket:
                    shopping_basket.remove('')
                record = frozenset(shopping_basket)
                yield record


if __name__ == "__main__":

    inFile = dataFromFile('association_mining.csv')
    minSupport = 0.2
    minConfidence = 0.6
    items, rules = runApriori(inFile, minSupport, minConfidence)

    saveResults(items, rules)
