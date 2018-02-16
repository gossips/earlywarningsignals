import unittest

class TestTimeSeries(unittest.TestCase):
    
    def test_fake_true(self):
        self.assertTrue(True)
        
    def test_fake_false(self):
        self.assertTrue(False)

if __name__ == '__main__':
    unittest.main()