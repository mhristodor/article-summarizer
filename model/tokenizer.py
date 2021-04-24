import unicodedata

class Tokenizer(object):
    
    def __init__(self, activate_lower_case = True):
        self.lower_case = activate_lower_case
        self.whiteChar = [" ", "\n", "\t", "\r"]
        self.punctuation = [(33,47),(58,64),(91,96),(123,126)]
    
    def specialChar(self,char):
        if char in self.whiteChar:
            return False
        
        if unicodedata.category(char).startswith("C"):
            return True

        return False


    def whitespace(self,char):
        if char in self.whiteChar:
            return True

        if unicodedata.category(char) == "Zs":
            return True

        return False

    def whitespaceTokenize(self,text):

        text = text.strip()
        
        if not text:
            return list()

        tokens = text.split()
        return tokens



    def punct(self,char):

        ordchr = ord(char)

        if True in [ordchr>=a and ordchr<=b for (a,b) in self.punctuation]:
            return True

        if unicodedata.category(char).startswith("P"):
            return True

        return False

    def cleanup(self,text):
        out = []

        for char in text:
            
            if ord(char) == 0 or self.specialChar(char):
                continue

            if self.whitespace(char):
                out.append(" ")
            else:
                out.append(char)    
        
        return "".join(out)

    def splitByPunct(self,text):
        chrs = list(text)
        out = []
        word = True

        for char in chrs:
            
            if self.punct(char):
                out.append([char])
                word = True
            
            else: 
                if word:
                    out.append([])
                
                word = False
                out[-1].append(char)

        return ["".join(_) for _ in out]

    def runTokenizer(self,text):
        
        text = self.cleanup(text)
        first = self.whitespaceTokenize(text)

        splits = []

        for token in first:
            if self.lower_case:
                token = token.lower()
            splits.extend(self.splitByPunct(token))

        return self.whitespaceTokenize(" ".join(splits))