from stats import mean, variance, normalize

def test_mean():
    assert mean([2,4,6]) == 4
    
def test_variance():
    assert round(variance([2,4,6]), 2) == 2.67
    
def test_normalize():
    result = normalize([2,4,6])
    # prumer normalizovanych dat ma byt 0 (v toleranci kvuli zaokrouhleni)
    assert round(sum(result)/len(result), 5) == 0