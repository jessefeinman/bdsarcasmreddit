import re
from functools import reduce
from re import search, findall
from re import sub
from string import punctuation

from nltk import ngrams, word_tokenize, pos_tag, FreqDist, ne_chunk, Tree
from nltk.corpus import opinion_lexicon
from nltk.corpus import wordnet, cmudict
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize.casual import _replace_html_entities

TWEET_LINK_RE = "https://t.co/(\w)+"
TWEET_HANDLE_RE = "@(\w)+"
HASHTAG_RE = "#(\w)+"
PUNCTUATION_RE = "[\'\!\"\#\$\%\&\/\(\)\*\+\,\-\.\/\:\;\<\=\>\?\@\[\]\^\_\`\{\}\|\~]"
REDDIT_USERNAMESUBREDDIT_RE = "(\/[ur]\/[a-z-A-Z0-9_-]+)"
REDDIT_LINK_RE = "(\[?(https?:\/\/(www\.)?)?[-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)\]?)"

lemma = WordNetLemmatizer()
negWords = frozenset(opinion_lexicon.negative())
posWords = frozenset(opinion_lexicon.positive())
suffixes = list(line.rstrip() for line in open("suffixes.txt"))
vader = SentimentIntensityAnalyzer()
d = cmudict.dict()

chunk = lambda posTagged: ne_chunk(posTagged, binary=True)
freq = lambda grams: FreqDist(grams)
lemmatize = lambda posTokens: [processPosTagsAndLemmatize(*wordPos) for wordPos in posTokens]
grams = lambda tokens, n: list(ngrams(tokens, n))
pos = lambda tokens: pos_tag(tokens)
posTagOnly = lambda tagged: [tag for word, tag in tagged]
tokenize = lambda text: word_tokenize(text)
wordCases = lambda ulls: [wordCase(*ull) for ull in ulls]
tokPosToTok = lambda listTokPos: list(list(zip(*listTokPos)))
tokNoNE = lambda chunked, removeNumbers=False: tokPosToTok(removeNamedEntities(chunked, removeNumbers))


def tokenizeFindAllRegex(r):
    regex = re.compile(r)
    return lambda s: re.findall(regex, s)


def capLetterFreq(ull):
    d = reduce(lambda i, l: i + l[1], ull, 0)
    return reduce(lambda i, u: i + u[0], ull, 0) / (d if d != 0 else 1)


def processPosTagsAndLemmatize(word, pos):
    return lemma.lemmatize(word, treebankToWordnetPOS(pos))


def removeNamedEntities(chunked, removeNumbers=True):
    def rec(t, r):
        if type(t) == Tree:
            if t.label() == 'NE':
                r.append(('NameTOK', '[NE]'))
            else:
                for child in t:
                    r.extend(rec(child, []))
        else:
            if removeNumbers:
                r.append(('NumTok', '[CD]')) if t[1] == "CD" else r.append(t)
            else:
                r.append(t)
        return r

    return rec(chunked, [])


def sentimentGrams(grams):
    return [{"LiuHu": sentimentLiuHu(gram), "Vader": sentimentVader(gram)} for gram in grams]


def syllableGrams(tokens, n):
    return grams([numSyllables(token) for token in tokens if token in d], n)


def vowelGrams(tokens, n):
    return grams([hasVowel(token) for token in tokens], n)


def sentimentLiuHu(gram):
    posWord = lambda word: word in posWords
    negWord = lambda word: word in negWords
    countPosNeg = lambda pn, word: (pn[0] + 1,
                                    pn[1],
                                    pn[2]) if posWord(word) else (pn[0],
                                                                  pn[1] + 1,
                                                                  pn[2]) if negWord(word) else (pn[0],
                                                                                                pn[1],
                                                                                                pn[2] + 1)
    p, n, u = reduce(countPosNeg, gram, [0, 0, 0])
    l = p + n + u
    return {"compound": round((p - n) / l, 3),
            "neg": round(n / l, 4),
            "neu": round(u / l, 4),
            "pos": round(p / l, 4)} if l > 0 else {"compound": 0,
                                                   "neg": 0,
                                                   "neu": 0,
                                                   "pos": 0}


def sentimentVader(gram):
    return vader.polarity_scores(" ".join(gram))


def tokenSuffixes(tokens):
    return [max(s2, key=len) for s2 in
            [s1 for s1 in
             [
                 [s0 for s0 in suffixes if word.endswith(s0)]
                 for word in tokens]
             if s1]]


def treebankToWordnetPOS(treebankPosTag):
    return {'J': wordnet.ADJ,
            'V': wordnet.VERB,
            'N': wordnet.NOUN,
            'R': wordnet.ADV}.get(treebankPosTag[0], wordnet.NOUN)


def upperLowerLen(tokensOriginalCase):
    return [
        (
            sum([1 if letter.isupper() else 0 for letter in token]),
            sum([1 if letter.islower() else 0 for letter in token]),
            len(token),
            token
        ) for token in tokensOriginalCase]


def wordCase(upper, lower, length, token):
    if upper == 0:
        return "NC"  # No Caps
    elif upper == length:
        return "AC"  # All Caps
    elif upper == 1 and token[0].isupper():
        return "FC"  # First Cap
    else:
        return "MC"  # Mixed Caps


def numSyllables(word):
    return [len(list(y for y in x if y[-1].isdigit())) for x in d[word.lower()]][0]


def hasVowel(word):
    for char in word.lower():
        if char in ['a', 'e', 'i', 'o', 'u']:
            return 1
    return 0


def hasPunctuation(word):
    return search(PUNCTUATION_RE, word) is not None


def punctuationFeatures(s):
    """
    s: input string
    returns dictionary of features (puncutation defined by string.punctuation)

    example:
    punctuation_features("Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed consequat magna eu facilisis!!?")
    {
     TOTAL: 5,
     TOTAL/LENGTH: 0.0543
     !_RAW: 2,
     !_RAW/LEN: 0.0217,
     !_RAW/TOTAL PUNCT FOUND: 0.4,
     ...
     @_RAW: 0.0,
     @_RAW/LEN: 0.0,
     @_RAW/TOTAL PUNCT FOUND: 0.0
    }
     """

    punctuation_dict = {}
    punctuation_found_list = findall(PUNCTUATION_RE, s)
    string_len = len(s)
    total_punctuation_found = len(punctuation_found_list)

    punctuation_dict["TOTAL"] = total_punctuation_found
    punctuation_dict["TOTAL/LENGTH"] = round(total_punctuation_found / string_len, 4)

    for p in punctuation:
        if p in punctuation_found_list:
            n = punctuation_found_list.count(p)
            punctuation_dict["{}_{}".format(p, "RAW")] = n
            punctuation_dict["{}_{}".format(p, "RAW/LEN")] = round(n / string_len, 4)
            punctuation_dict["{}_{}".format(p, "RAW/TOTAL_PUNCT_FOUND")] = round(n / total_punctuation_found, 4)
        else:
            punctuation_dict["{}_{}".format(p, "RAW")] = 0.
            punctuation_dict["{}_{}".format(p, "RAW/LEN")] = 0.
            punctuation_dict["{}_{}".format(p, "RAW/TOTAL_PUNCT_FOUND")] = 0.

    return punctuation_dict


def keyToStr(d, name=""):
    new = {}
    for key, value in d.items():
        new_key = name + ' ' + str(" ".join([str(k) for k in key]))
        new[new_key] = value
    return new


def stripTweet(tweet):
    tweet = sub(TWEET_HANDLE_RE, "NameTOK", tweet)
    tweet = sub(TWEET_LINK_RE, "LinkTOK", tweet)
    tweet = sub(HASHTAG_RE, "", tweet)
    return tweet


def unicodeReplacement(tweet):
    return _replace_html_entities(tweet)


def partialSentimentFeat(tokens):
    sent = {
        'fullSent': sentimentGrams([tokens]),
        'halfSent1': sentimentGrams([tokens[:int(len(tokens) / 2)]]),
        'halfSent2': sentimentGrams([tokens[int(len(tokens) / 2):]]),
        'thirdSent1': sentimentGrams([tokens[:int(len(tokens) / 3)]]),
        'thirdSent2': sentimentGrams([tokens[int(len(tokens) / 3):2 * int(len(tokens) / 3)]]),
        'thirdSent3': sentimentGrams([tokens[2 * int(len(tokens) / 3):]]),
        'quartSent1': sentimentGrams([tokens[:int(len(tokens) / 4)]]),
        'quartSent2': sentimentGrams([tokens[int(len(tokens) / 4):2 * int(len(tokens) / 4)]]),
        'quartSent3': sentimentGrams([tokens[2 * int(len(tokens) / 4):3 * int(len(tokens) / 4)]]),
        'quartSent4': sentimentGrams([tokens[3 * int(len(tokens) / 4):]])
    }
    vader = {}
    liu = {}
    for key, val in sent.items():
        vader[key + "Vader"] = (val[0]['Vader']['compound'] + 1) / 2
        liu[key + "LiuHu"] = (val[0]['LiuHu']['compound'] + 1) / 2
    return vader, liu


def suffixesFeat(tokens):
    sufftok = tokenSuffixes(tokens)
    suffreq = dict(freq(sufftok))
    normSuffFreq = {}
    norm2SuffFreq = {}
    sumSuf = sum(suffreq.values())
    for key, val in suffreq.items():
        normSuffFreq[key] = val / sumSuf
        norm2SuffFreq[key] = val / len(tokens)
    normSuffFreq = keyToStr(normSuffFreq, name='normSuffFreq')
    norm2SuffFreq = keyToStr(norm2SuffFreq, name='normSuffFreq')
    return sufftok, normSuffFreq, norm2SuffFreq


def tokPosTagsNoNE(tokens):
    tagged = pos(tokens)
    chunked = chunk(tagged)
    (tokens, postags) = tokNoNE(chunked)
    return tokens, postags


def casesFeat(tokens):
    ull = upperLowerLen(tokens)
    cases = wordCases(ull)
    capFreq = capLetterFreq(ull)
    allCapsFreq = cases.count('AC') / len(cases)
    return capFreq, allCapsFreq


def cleanTokensTwitter(tweet):
    tweet = stripTweet(tweet)
    tweet = unicodeReplacement(tweet)
    tokens = tokenize(tweet)
    return tokens, tweet


def gramFreqFeat(gramFun, tok, minGram, maxGram, name):
    d = {}
    for g in range(minGram, maxGram + 1):
        grams = gramFun(tok, g)
        freqTable = dict(freq(grams))
        n = str(name + str(g))
        dic = keyToStr(freqTable, str(name + str(g)))
        d[n] = dic
    return d

def stripReddit(tweet):
    tweet = sub(REDDIT_USERNAMESUBREDDIT_RE, "NameTOK", tweet)
    tweet = sub(REDDIT_LINK_RE, "LinkTOK", tweet)
    return tweet

list_re = [
    r"(\/sarcasm)",
    r"(<\/?sarcasm>)",
    r"(&lt;\/?sarcasm&gt)",
    r"(#sarcasm)",
    r"( \/s(?![a-zA-Z0-9]))",
    r"(:\^(?! ?[D)(\[\]]))"
          ]
sarcasm_re = re.compile('|'.join(list_re))
que = re.compile(r"(\[\?\](?!\())")
exc = re.compile(r"(\[!\](?!\())")
period = re.compile(r"(\.~)|(\. ~)")

def stripSarcasm(tweet):
    tweet = sarcasm_re.sub('', tweet)
    tweet = que.sub('?', tweet)
    tweet = exc.sub('!', tweet)
    tweet = period.sub('.',tweet)
    return tweet

def cleanTokensReddit(tweet):
    tweet = stripReddit(tweet)
    tweet = stripSarcasm(tweet)
    tweet = unicodeReplacement(tweet)
    tokens = tokenize(tweet)
    return tokens, tweet

def feature(text, cleanTokens):
    text = repr(text)
    (tokens, text) = cleanTokens(text)

    (tokens, postags) = tokPosTagsNoNE(tokens)
    (capFreq, allCapsFreq) = casesFeat(tokens)

    puncuationFreq = punctuationFeatures(text)
    (vader, liu) = partialSentimentFeat(tokens)
    (sufftok, normSuffFreq, norm2SuffFreq) = suffixesFeat(tokens)
    grm = gramFreqFeat(grams, tokens, 1, 3, 'grm')
    syl = gramFreqFeat(syllableGrams, tokens, 1, 4, 'syl')
    vow = gramFreqFeat(vowelGrams, tokens, 1, 4, 'vow')
    pos = gramFreqFeat(grams, postags, 1, 4, 'pos')
    suf = gramFreqFeat(grams, sufftok, 1, 4, 'suf')
    return {**grm,
            **syl,
            **vow,
            **pos,
            **suf,
            'puncuationFreq': puncuationFreq,
            'normSuffFreq': normSuffFreq,
            'norm2SuffFreq': normSuffFreq,
            'sentimentVader': vader,
            'sentimentLiu': liu,
            'capFreq': capFreq
            }
