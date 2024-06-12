import unittest
from pathlib import Path

from rna_sdb.utils import parse_alignment, wuss_to_db


class TestWussToDb(unittest.TestCase):
    def test_wuss_to_db_1(self):
        # Case: dot-bracket structure
        structure = "((((...((((.......(((..........)))........)))).....))))"
        target = "((((...((((.......(((..........)))........)))).....))))"

        self.assertEqual(wuss_to_db(structure), target)

    def test_wuss_to_db_2(self):
        # Case: empty structure
        structure = ""
        target = ""

        self.assertEqual(wuss_to_db(structure), target)

    def test_wuss_to_db_3(self):
        # Case: example from URS0000D6C384_12908/1-78
        # https://rfam.org/family/RF03160/alignment?acc=RF03160&format=stockholm&download=0
        structure = "((((---((((BB<<<<<<__......___......_>>>>>><<<__AAA__bb_>>>,,,,,.))))--aaa))))"
        target = "((((...((((..((((((..................))))))(((..........)))......)))).....))))"

        self.assertEqual(wuss_to_db(structure), target)

    def test_wuss_to_db_4(self):
        # Case: example from URS0000D669BF_12908/1-56
        # https://rfam.org/family/RF03160/alignment?acc=RF03160&format=stockholm&download=0
        structure = "((((---((((BB_____<<<__AAA__bb_>>>,,,,,...))))--aaa.))))"
        target = "((((...((((.......(((..........)))........))))......))))"

        self.assertEqual(wuss_to_db(structure), target)

    def test_wuss_to_db_5(self):
        # Case: example from URS0000D6C67E_12908
        # https://rfam.org/family/RF03160/alignment?acc=RF03160&format=stockholm&download=0
        structure = "((((---((((BB_____<<<__AAA__bb_>>>,,,,,...))))--aaa))))"
        target = "((((...((((.......(((..........)))........)))).....))))"

        self.assertEqual(wuss_to_db(structure), target)


class TestParseAlignment(unittest.TestCase):
    def setUp(self):
        self.data_dir = Path(__file__).parent.resolve() / "data"

    def test_parse_alignment_1(self):
        # Case: example from RF03160
        # https://rfam.org/family/RF03160#tabview=tab2
        file_path = self.data_dir / "RF03160.stockholm.txt"

        df, rfam_id, ss_cons = parse_alignment(file_path)

        self.assertEqual(rfam_id, "RF03160")
        self.assertEqual(
            ss_cons,
            "((((.---.((((BB<<<<<<__.................._____........................................................_>>>>>><<<__AAA__bb._>>>................,,,,,........................)))).--aaa....))))",
        )
        self.assertEqual(len(df), 1613)  # number of seed alignment sequences

        item = df[df["seq_name"] == "URS0000D6C384_12908/1-78"].iloc[0]
        self.assertEqual(
            item["seq"],
            "UUUUUAACCCAGCCACUAGCAUUGACACUUUGUCGUGUUCGUGCCGGUCCCAAGCCCGGAGAAAAUGGGGAGGUUUUU",
        )
        self.assertEqual(
            item["db_structure"],
            "((((...((((..((((((..................))))))(((..........)))......)))).....))))",
        )

        item = df[df["seq_name"] == "URS0000D669BF_12908/1-56"].iloc[0]
        self.assertEqual(
            item["seq"],
            "GGCCUAAUGCAGCAUAGUCCUGUCACAAGCCAGGCUGAAAAAUGCAGAGUGAGGCA",
        )
        self.assertEqual(
            item["db_structure"],
            "((((...((((.......(((..........)))........))))......))))",
        )

        item = df[df["seq_name"] == "URS0000D6C67E_12908/1-55"].iloc[0]
        self.assertEqual(
            item["seq"],
            "CUCCUAAUGCAGCCGAAGGCGGUCACAAGCCCGAUUGAGAGAUGCAGAGUGGGAA",
        )
        self.assertEqual(
            item["db_structure"],
            "((((...((((.......(((..........)))........)))).....))))",
        )

    def test_parse_alignment_2(self):
        # Case: example from RF02913
        # https://rfam.org/family/RF02913#tabview=tab2
        file_path = "data/RF02913.stockholm.txt"

        df, rfam_id, ss_cons = parse_alignment(file_path)

        self.assertEqual(rfam_id, "RF02913")
        self.assertEqual(
            ss_cons,
            ":::::::::::(<<<<.._______________________________________________________________________________________..>>>>,,<<<-<<<______.----->>>->>>,<<<__________>>>):::",
        )
        self.assertEqual(len(df), 1542)  # number of seed alignment sequences

        # item = df[df["seq_name"] == "URS0000D67E4D_12908/1-58"].iloc[0]
        # self.assertEqual(
        #     item["seq"],
        #     "AUAAUGAUACUUCCCUAUGGGGCUGGCGGAGAACCCGGCAGAGGUGAAAUCCCUAUGA",
        # )
        # self.assertEqual(
        #     item["db_structure"],
        #     "............((((()))))(((.(((.....))).)))(((.......)))....",
        # )

        # item = df[df["seq_name"] == "URS0000D6A202_12908/1-70"].iloc[0]
        # self.assertEqual(
        #     item["seq"],
        #     "AUAAUGAUACUUCCGUCUCGGUUUACCGGGGCGGCUCGGAGCGAUCCUCGGAGGGGUGAGAGUCCCAUGA",
        # )
        # self.assertEqual(
        #     item["db_structure"],
        #     "............((((((..........))))))(((.(((.....))).)))(((.......)))....",
        # )

        # item = df[df["seq_name"] == "URS0000D68C3F_12908/1-96"].iloc[0]
        # self.assertEqual(
        #     item["seq"],
        #     "AUAAUGAUACUUCUGUCCAGCCACAGCCCCUGCACUGCGGGGAUUACGCAAUGGGGGCAGCCCGGUGAGACCCACGGGGAGGUUAGAUUCCUAUGG",
        # )
        # self.assertEqual(
        #     item["db_structure"],
        #     "............((((((....................................))))))(((.(((.....))).)))(((.......)))....",
        # )

    def test_parse_alignment_3(self):
        # Case: example from RF02924
        # https://rfam.org/family/RF02924#tabview=tab2
        file_path = "data/RF02924.stockholm.txt"

        df, rfam_id, ss_cons = parse_alignment(file_path)

        self.assertEqual(rfam_id, "RF02924")
        self.assertEqual(
            ss_cons,
            "........................................................>>>>>>...............................................................----->..,,,,<<.<<<.......................___aaaaa_.>>>>>...))).)))))",
        )
        self.assertEqual(len(df), 1426)  # number of seed alignment sequences

        item = df[df["seq_name"] == "URS0000D6C384_12908/1-78"].iloc[0]
        self.assertEqual(
            item["seq"],
            "UAAGUUAGAAGUUGAAAUAGUUGUUAUUAAUUAAUGAUGCGGUCCCAACGCAUCAAGCCCUAAUUGGGAGGUGAUAAGUGAUGUGGGGGGUGGCAGUCCCACCUAACCUA",
        )
        self.assertEqual(
            item["db_structure"],
            "((((((((..(((((.............................))))).(((((.............)))).....)....(((((..........)))))))))))))",
        )

        item = df[df["seq_name"] == "URS0000D669BF_12908/1-56"].iloc[0]
        self.assertEqual(
            item["seq"],
            "GGCCUAAUGCAGCAUAGUCCUGUCACAAGCCAGGCUGAAAAAUGCAGAGUGAGGCA",
        )
        self.assertEqual(
            item["db_structure"],
            "((((...((((.......(((..........)))........))))......))))",
        )

        item = df[df["seq_name"] == "URS0000D6C67E_12908/1-55"].iloc[0]
        self.assertEqual(
            item["seq"],
            "CUCCUAAUGCAGCCGAAGGCGGUCACAAGCCCGAUUGAGAGAUGCAGAGUGGGAA",
        )
        self.assertEqual(
            item["db_structure"],
            "((((...((((.......(((..........)))........)))).....))))",
        )


class TestParseAlignment_2(unittest.TestCase):
    """Test parsing alignment files generated with Infernal cmalign"""

    def setUp(self):
        self.data_dir = Path(__file__).parent.resolve() / "data"

    def test_parse_alignment_1(self):
        file_path = self.data_dir / "RF00040.sto"

        df, rfam_id, ss_cons = parse_alignment(file_path)

        self.assertIsNone(rfam_id)

        self.assertEqual(
            ss_cons,
            "::<<<<<<..-----....<<<<<....____~~~~~~~~~~~~~~~~~~~~....>>>>>----...>>>>>>,,,,..................................................................................,,,,,...<<<<<<<<---<<<<---.<<<--<<____>>------>>>.-->>>>--.->>>>>>>>.-----(((((((((((------.-(((((((,,,,,...<<<<<.--.-.-<<<<<___________~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~..>>>>>-.---..>>>>>.,.,,.,,,...,,<<<-<<<<-<<.<<._____~~~~~~~~~~~~~~~~~~~~~~~~~~.>>>>.->>>>>>>,<..<<<<..---..-----<<<<<.-.-.-.---..--<<-<<<<<____>>>>>->>------..---------->>>>>-->>>>>,,,.,,,.)))))))..--------.......---.)))).)..))).)).)::.:::::::",  # noqa F401
        )

        seq = "CUUUAUAUGGCAGUUGAAAGAUAAGGGUGUAAUACGAAAUUUUGCCGGAUAGCCCAUACACGCAGCAAUGGCGUAAGACGUAUUGUGCAAUCAGGCAUUUAGCGGGCUGCGGGUUGCAGCCUGGACGAAAACGGGAUUGGUUCACCUGGUAUUAUACUGGCGGUUCUCCCUAUCAAAGCGCUGUUUUUCCAUAUGUAAAACAGGCUACCGAGUAUUUGCGCCCCAAGAGCAGCCGACAUCCGUGAGGUUGACGGCUUUGCGAUGAGACACGGGGUCAUCGGUCUUUCACGUCCAGCGUUACUUUGCCCGCAGCUUUGUCACUAA"  # noqa F401
        self.assertEqual(
            str(
                df[df["seq_name"] == "JMSO01000136.1/4751-5074"]["seq"].iloc[0]
            ).upper(),
            seq,
        )

        structure = "::<<<<<<---<.<<____>>>---->>>>>>,,,,,,,,,<<<<<<<<---<<<<---<<<--<<____>>------>>>-->>>>--->>>>>>>>-----(((((((((((-------((((((,,,,,<<<<<----<<<<<___________.>>>>>>>>>>,,,,,,,,<<<-<<<<-<<<<_____>>>>->>>>>>><<<<<--------<<<<<---.-----<<-<<<<<____>>>>>->>------.---------->>>>>-->>>>>,,,,,,))))))----------))))))))))):::::::::"  # noqa F401
        self.assertEqual(
            str(df[df["seq_name"] == "JMSO01000136.1/4751-5074"]["structure"].iloc[0]),
            structure,
        )

    def test_parse_alignment_2(self):
        file_path = self.data_dir / "RF02966.sto"

        df, rfam_id, ss_cons = parse_alignment(file_path)

        self.assertIsNone(rfam_id)
        self.assertEqual(
            ss_cons,
            "::::::[[[[[[[[[[,,,,,,,,((.((.<<<<<...<<______.__~~~~~>>>>>>>,,,...,<<<.<<<<_____...>>>>>>>,,.)).)),,,,<<<...-<<<<...___..____~~~~~~~~~~~~~~~~~~~~~~~~~~~~~......>>>>-.>>>,,.,,]].]]]]].]]]",  # noqa F401
        )

        # Compare against 2D structures from RNAcentral
        test_cases = [  # seq_name: (seq_name, structure from RNAcentral)
            (
                "URS0000D68B29_1263000/1-123",
                "......((((((((((........(((((((((((........)))))))....(((((((.....)))))))..))))....(((..((((.........))))))).....))))))))))",  # noqa F401
            ),
            (
                "URS0000D67B42_12908/1-117",
                ".....((((((((((........(((((((((((........)))))))....(((((((...)))))))..))))....(((..((((.......)))))))....))))))))))",  # noqa F401
            ),
            (
                "URS0000D69C6A_12908/1-118",
                "......((((((((((........(((((((((((........)))))))....(((((((...)))))))..))))....(((..((((.......)))))))....))))))))))",  # noqa F401
            ),
            (
                "KE159703.1/781287-781403",
                "......((.(((((((........(((((((((((......)))))))....(((((((.....)))))))..))))....(((.((((.......)))).)))....)))))))))",  # noqa F401
            ),
            (
                "FR893678.1/3771-3642",
                "......((((((((((........((.(((((((((........))))))).....(((((((...)))))))..)).))....(((...(((..............)))..))).....))))))))))",  # noqa F401
            ),
            (
                "AAXG02000032.1/4729-4615",
                "......((.(((((((........(((((((((((......)))))))....(((((((....)))))))..))))....(((.((((......)))).)))....)))))))))",  # noqa F401
            ),
            (
                "URS0000D693A8_1235797/1-117",
                "......((.(((((((........(((((((((((......)))))))....(((((((.....)))))))..))))....(((.((((.......)))).)))....)))))))))",  # noqa F401
            ),
            (
                "URS0000D6BB32_12908/1-115",
                "......((.(((((((........(((((((((((......)))))))....(((((((....)))))))..))))....(((.((((......)))).)))....)))))))))",  # noqa F401
            ),
            (
                "URS0000D68FC6_411467/1-115",
                "......((.(((((((........(((((((((((......)))))))....(((((((....)))))))..))))....(((.((((......)))).)))....)))))))))",  # noqa F401
            ),
            (
                "URS0000D69E8B_12908/1-123",
                "......(((..(((((........(((((((((((........)))))))....(((((((....)))))))..))))....(((....((((........))))..))).....))))))))",  # noqa F401
            ),
            (
                "URS0000D6D059_12908/1-123",
                "......(((..(((((........(((((((((((........)))))))....(((((((....)))))))..))))....(((....((((........))))..))).....))))))))",  # noqa F401
            ),
            (
                "URS0000D67468_12908/1-116",
                "......((((((((((........(((((((((((........)))))))....(((((((.....)))))))..))))....((((.......)...))).....))))))))))",  # noqa F401
            ),
            (
                "URS0000D66339_1262989/1-130",
                "......((((((((((........((.(((((((((........))))))).....(((((((...)))))))..)).))....(((...(((..............)))..))).....))))))))))",  # noqa F401
            ),
            (
                "URS0000D66339_12908/1-130",
                "......((((((((((........((.(((((((((........))))))).....(((((((...)))))))..)).))....(((...(((..............)))..))).....))))))))))",  # noqa F401
            ),
            (
                "URS0000D6B2F0_12908/1-120",
                "......(((..(((((........(((((((((((........)))))))....(((((((.....)))))))..))))....(((..((((.......)))).))).....))))))))",  # noqa F401
            ),
            (
                "URS0000D680B7_12908/1-116",
                "......((((((((((........(((((((((((........)))))))....(((((((.....)))))))..))))....((((.......)...))).....))))))))))",  # noqa F401
            ),
            (
                "URS0000D658E2_12908/1-120",
                "......(((((((((........(((((((((((........)))))))....(((((((...)))))))..))))....(((.((((.........)))).)))....)).))))).))",  # noqa F401
            ),
        ]

        for seq_name, structure in test_cases:
            structure_parsed = df[df["seq_name"] == seq_name].iloc[0]["db_structure"]
            self.assertEqual(structure_parsed, structure)


if __name__ == "__main__":
    unittest.main()
