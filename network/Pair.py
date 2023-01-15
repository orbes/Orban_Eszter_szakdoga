# egy élt megvalósító osztály
# két él ugynaz, ha ugyanazok a nodok a cél és startnodejaik
# ((egyik célja = másik célja és egyik start = másik start)
#   vagy (egyik cél = másik start és egyik start =  másik cél))

class Pair:
    def __init__(self, start, destination):
        self.start = start
        self.destination = destination

    def __eq__(self, other):
        if not isinstance(other, Pair):
            return NotImplemented
        return (self.start == other.start and self.destination == other.destination) or (
                    self.start == other.destination and self.destination == other.start)

    def __repr__(self):
        return str(self.start) + ': ' + self.destination
