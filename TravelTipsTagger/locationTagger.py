import sys
sys.path.append("./word_utils")
import pandas
from pypinyin import lazy_pinyin
import jieba
from jieba import analyse
from jieba import posseg as pseg
from word_utils.langconv import Converter
import json
import re
import os, pickle
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial
import argparse

r"""对单个文本进行清洗"""
def clean_content(content:str)->str:
    # 仅保留文本中的简体和繁体字
    cleanRule = re.compile(u'[^\u4e00-\u9fff]+')
    content = re.sub(cleanRule, "", str(content))
    content = "".join(content.split()) #主要为了去除\xao,即&nbsp
    # 繁体转简体
    content = Converter("zh-hans").convert(content)
    return content

r"""城镇信息节点，保存城镇名和其所有直属上级行政单位"""
class TownInfo:
    def __init__(self, townName):
        self.townName = townName
        self.fatherTown = [] # 为列表是因为存在同名城市的情况，需保存其所有可能的上级行政单位

r"""读取城镇信息CSV文件，依次搭建城镇信息节点映射表并序列化输出到指定路径文件"""
def build_townInfo(csvPath:str, savePath:str)->dict:
    if os.path.exists(savePath):
        with open(savePath, 'rb') as f:
            townNameMap = pickle.load(f)
        return townNameMap
    townNameMap = dict() # 城镇名与对应节点实例哈希映射表
    csvTownInfo = pandas.read_csv(csvPath)
    for idx in tqdm(csvTownInfo['id'], desc="构建城镇信息节点映射表", ncols=150):
        townName = csvTownInfo[csvTownInfo['id']==idx]['name'].item()
        townName = townName.strip()
        townInfo = TownInfo(townName)
        if not townName in townNameMap.keys():
            townNameMap[townName] = townInfo
        else:
            townInfo = townNameMap[townName]
        # 遍历各城市行政级别，若为空或同名（自身即属该级）则继续向上寻找，否则停止
        for level in ['street','area','city','province']:
            fatherTownName = csvTownInfo[csvTownInfo['id']==idx][level].item()
            if pandas.isna(fatherTownName) or fatherTownName.strip()==townInfo.townName:
                continue
            else:
                fatherTownName = fatherTownName.strip()
                if not fatherTownName in townNameMap.keys():
                    fatherTownInfo = TownInfo(fatherTownName)
                    townInfo.fatherTown.append(fatherTownInfo)
                else:
                    fatherTownInfo = townNameMap[fatherTownName]
                    townInfo.fatherTown.append(fatherTownInfo)
                break
    
    # 根据城镇名的拼音小写首字母构造二级索引映射
    # 因此townNameMap最终形如：{'b':{'北京':TownInfo instance of 北京, ...}}
    newTownNameMap = dict()
    for townName, townInfo in townNameMap.items():
        firstLatter = lazy_pinyin(townName[0])[0][0]
        if not firstLatter in newTownNameMap.keys():
            newTownNameMap[firstLatter] = {townName: townInfo}
        else:
            newTownNameMap[firstLatter][townName] = townInfo
    townNameMap = newTownNameMap
    with open(savePath, 'wb') as f:
        pickle.dump(townNameMap, f)
    return townNameMap

r"""从单个文本中提取城镇信息的方法1"""
def tag_content_v1(content:str, townNameMap:dict)->dict:
    """
    思路：利用tf-idf和textRank算法分别提取文本top15关键词并去交集，对关键词进行词性标注判别，取地名关键词与城镇名表匹配
    优点：能提取出和文章内容最相关的城镇名，建议使用本方法
    缺点：不能保证每篇文章都提取出城镇名
    """
    tags1 = analyse.extract_tags(content, topK=15)
    tags2 = analyse.textrank(content, topK=15)
    tags = [each for each in set(tags1+tags2) if each in tags1 and each in tags2]
    townTags = dict()
    for tag in tags:
        if tag == "中国":
            continue
        tagFirstLatter = lazy_pinyin(tag[0])[0][0]
        for townName in townNameMap[tagFirstLatter].keys():
            if townName.find(tag) == 0: # 必须要从从第一个字开始匹配上的才行
                _, flag = pseg.lcut(tag, use_paddle=True)[0]
                if flag == 'LOC' or flag == 'ns':
                    townTags[townName] = content.count(tag.strip())
                break
    return townTags

r"""从单个文本中提取城镇信息的方法2"""
def tag_content_v2(content:str, townNameMap:dict)->dict:
    """
    思路：直接对文本进行分词并对所有词进行词性标注，筛选出词性为地名的词，再和城镇名表匹配
    优点：保证所有文章都能提取出包含在其中的可能城镇名
    缺点：一篇文章会提取出众多不相关的城镇名，对长文本的词性标注效率低
    """
    words = pseg.cut(content, use_paddle=True)
    locWords = []
    for word, flag in words:
        if flag=='LOC' or flag=='ns':
            locWords.append(word)
    townTags = dict()
    
    for word in set(locWords):
        wordFirstLetter = lazy_pinyin(word[0])[0][0]
        for townName in townNameMap[wordFirstLetter].keys():
            if townName.find(word) == 0:
                townTags[townName] = content.count(word.strip())
                break
    return townTags

r"""对初步提取出的文本相关城镇信息，对照城镇节点映射表，筛选出最相关的一个或多个城镇信息"""
def town_filter(townTags:dict, townNameMap:dict)->dict:
    if not townTags.keys():
        return townTags
    belongCandidates = dict() # 保存候选省份/直辖市的共现频率
    townBelongTowns = dict() # 保存每个城镇可能所属的省份/直辖市
    # 遍历各城镇获取其所属省份/直辖市并保存相关信息到以上两个字典中
    for townName in townTags.keys():
        firstLatter = lazy_pinyin(townName[0])[0][0]
        townInfo = townNameMap[firstLatter][townName] # 从映射表中取出节点实例
        belongTowns = _to_father(townInfo)
        townBelongTowns[townName] = belongTowns
        for belongTown in belongTowns:
            if not belongTown in belongCandidates.keys():
                belongCandidates[belongTown] = townTags[townName]
            else:
                belongCandidates[belongTown] += townTags[townName]
    # 筛选可能性最大的省份/直辖市作为旅游攻略所针对城市所在的地区
    mostProbArea = max(belongCandidates, key=belongCandidates.get)
    return {name: townTags[name] for name in townTags.keys() if mostProbArea in townBelongTowns[name]}

r"""在town_filter方法中使用到的辅助方法，通过递归向前搜索单个城镇节点所对应的所有可能根节点"""
def _to_father(townInfo:TownInfo)->list:
    # 递归向前搜索父节点直至找到城镇可能所属的省份或直辖市
    if not townInfo.fatherTown:
        return [townInfo.townName]
    father = []
    for fatherInfo in townInfo.fatherTown:
        father.extend(_to_father(fatherInfo))
    return father

r"""对单个文本进行清洗和信息提取的方法"""
def extract_one_content(idAndContent:tuple, townNameMap:dict)->list:
    id, raw = idAndContent
    # 清洗文本
    content = clean_content(raw)
    # 提取城镇信息,可以更换方法为tag_content_v2,但效率会变低
    townTags = tag_content_v1(content, townNameMap)
    # 筛选正确的城镇信息
    # 思路是利用哈希表和节点递归获取每个城镇所属省份/直辖市，基于文本共现频率来筛选出所属省份可能性最大的城镇
    townTags = town_filter(townTags, townNameMap)
    return [id, townTags]

r"""本模块的主方法，并行处理多个文本以提取其中的城镇信息"""
def extract_contents(allTownCsvPath, townNameMapSavePath, travelTipsPath, idfPath, outPath):
    # 一些初始化工作
    jieba.enable_paddle()
    analyse.set_idf_path(idfPath)
    # 构建城镇名信息节点哈希映射表和城镇名列表
    townNameMap = build_townInfo(allTownCsvPath, townNameMapSavePath)
    # 读取旅游攻略excel文件并新增一列保存后续提取的城镇信息
    travelTips = pandas.read_excel(travelTipsPath)
    travelTips['townInfo'] = [
        dict() for _ in range(len(travelTips['articleID']))
    ]
    # 对每一篇攻略，清洗文本并从中提取与之相关的城镇信息
    with Pool(max(1, cpu_count() - 2)) as p:  # 这里-2是因为作者本机用满全部核心存在卡死的风险，
                                              # 部署到服务器上可视情况把其去掉
        annotate_ = partial(
            extract_one_content,
            townNameMap=townNameMap
        )
        articleTagList = list(
            tqdm(
                p.imap(annotate_, zip(travelTips['articleID'], travelTips['contents']), chunksize=32),
                total=len(travelTips['articleID']),
                desc="提取文本城镇信息中",
                ncols=150,
            )
        )
    articleTag = dict()
    for pair in articleTagList:
        id, tags = pair
        articleTag[id] = tags
    # 保存结果到输出文件中
    if os.path.exists(outPath):
        os.remove(outPath)
    with open(outPath, 'w') as f:
        json.dump(articleTag, f)

if __name__ == '__main__':
    allTownCsvPath = './data_scripts/allTown.csv'
    townNameMapSavePath = './data_scripts/townNameMap.obj'
    travelTipsPath = './data_scripts/fuJian.xlsx'
    idfPath = './data_scripts/idf_allTown.txt'
    outPath = './outputs/fuJian_travelTips_townTags.json'
    extract_contents(allTownCsvPath, townNameMapSavePath, travelTipsPath, idfPath, outPath)
