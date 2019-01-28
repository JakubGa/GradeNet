import unittest
import net
import numpy

class Test_TestNet(unittest.TestCase):

    midis = list()
    file_list = {'ashover29.mid','ashover30.mid','NARDIS.mid','onenote.mid',
        'ragdoll.mid','schum_abegg.mid','scn15_11.mid','SilenceIsGolden.mid'}
    @classmethod
    def setUpClass(cls):
        for file in cls.file_list:
            cls.midis.append(net.to_numpy_array(net.read_midi_file(file,8)))

    @classmethod
    def tearDownClass(cls):
        pass

    def test_inting(self):
        for song in self.midis:
            int_array = net.array_to_int(song,58,16)
            self.assertTrue(numpy.array_equal(net.int_to_bool(int_array,58,16)[:,58:58+16], song[:,58:58+16]))

    def test_opening(self):
        for song in self.midis:
            self.assertIsNotNone(song)

    def test_to_numpy(self):
        pass



if __name__ == '__main__':
    unittest.main()