import unittest

from codeocean_assets import CodeOceanCatalog, AssetRecord, resolve_asset_pairs


RAW_ID = "ed332862-d8fb-4760-a04d-106e59a0cd8f"
SORTED_ID = "3afc8120-656c-4422-94ca-fbd6995f5b8e"
SESSION_ID = "853781_2026-07-01_12-20-31"


def raw_asset(**overrides):
    value = {
        "id": RAW_ID,
        "name": f"ecephys_{SESSION_ID}",
        "data_description": {
            "data_level": "raw",
            "subject_id": "853781",
            "creation_time": "2026-07-01T12:20:31-07:00",
        },
        "state": "ready",
    }
    value.update(overrides)
    return value


def sorted_asset(**overrides):
    value = {
        "id": SORTED_ID,
        "name": f"ecephys_{SESSION_ID}_sorted_2026-07-16_16-50-35",
        "data_description": {
            "data_level": "derived",
            "subject_id": "853781",
            "creation_time": "2026-07-16T16:50:35.848431Z",
            "source_data": [f"ecephys_{SESSION_ID}"],
        },
        "state": "ready",
    }
    value.update(overrides)
    return value


class AssetRecordTests(unittest.TestCase):
    def test_normalizes_aind_metadata(self):
        asset = AssetRecord.from_value(sorted_asset())

        self.assertEqual(SORTED_ID, asset.id)
        self.assertEqual("853781", asset.subject_id)
        self.assertEqual((f"ecephys_{SESSION_ID}",), asset.source_data)
        self.assertTrue(asset.is_sorted_ecephys)


class ResolveAssetPairsTests(unittest.TestCase):
    def test_resolves_known_853781_pair_from_lineage(self):
        resolution = resolve_asset_pairs(
            [sorted_asset(), raw_asset()], subject_id="853781"
        )

        self.assertEqual((), resolution.errors)
        self.assertEqual(1, len(resolution.pairs))
        pair = resolution.pairs[0]
        self.assertEqual(SESSION_ID, pair.session_id)
        self.assertEqual(RAW_ID, pair.raw.id)
        self.assertEqual(SORTED_ID, pair.sorted.id)
        self.assertEqual("source_data", pair.pairing_method)

    def test_selects_latest_sorting_by_default(self):
        older = sorted_asset(
            id="older-sorting",
            name=f"ecephys_{SESSION_ID}_sorted_2026-07-10_10-00-00",
            data_description={
                "data_level": "derived",
                "subject_id": "853781",
                "creation_time": "2026-07-10T10:00:00Z",
                "source_data": [f"ecephys_{SESSION_ID}"],
            },
        )

        resolution = resolve_asset_pairs(
            [raw_asset(), older, sorted_asset()], subject_id=853781
        )

        self.assertEqual(SORTED_ID, resolution.pairs[0].sorted.id)
        self.assertIn("older-sorting", resolution.warnings[0])

    def test_name_fallback_is_reported(self):
        without_lineage = sorted_asset(
            data_description={
                "data_level": "derived",
                "subject_id": "853781",
                "creation_time": "2026-07-16T16:50:35Z",
            }
        )

        resolution = resolve_asset_pairs(
            [raw_asset(), without_lineage], subject_id=853781
        )

        self.assertEqual("name_fallback", resolution.pairs[0].pairing_method)
        self.assertIn("no usable source_data lineage", resolution.warnings[0])


class FakeDataAssets:
    def __init__(self, assets):
        self.assets = assets
        self.search_params = None

    def search_data_assets_iterator(self, search_params):
        self.search_params = search_params
        yield from self.assets


class FakeClient:
    def __init__(self, assets):
        self.data_assets = FakeDataAssets(assets)


class MetadataCatalog(CodeOceanCatalog):
    def __init__(self, client, descriptions):
        super().__init__(client)
        self.descriptions = descriptions

    def _read_data_description(self, asset_id):
        return self.descriptions[asset_id]


class CodeOceanCatalogTests(unittest.TestCase):
    def test_enriches_search_results_before_pairing(self):
        raw = {
            "id": RAW_ID,
            "name": f"ecephys_{SESSION_ID}",
            "created": 1782948031,
            "state": "ready",
        }
        sorted_value = {
            "id": SORTED_ID,
            "name": f"ecephys_{SESSION_ID}_sorted_2026-07-16_16-50-35",
            "created": 1784249435,
            "state": "ready",
        }
        descriptions = {
            RAW_ID: raw_asset()["data_description"],
            SORTED_ID: sorted_asset()["data_description"],
        }
        client = FakeClient([raw, sorted_value])
        catalog = MetadataCatalog(client, descriptions)

        assets, warnings = catalog.search_subject_assets("853781")
        resolution = resolve_asset_pairs(assets, "853781")

        self.assertEqual((), warnings)
        self.assertEqual(RAW_ID, resolution.pairs[0].raw.id)
        self.assertEqual(SORTED_ID, resolution.pairs[0].sorted.id)
        search_filter = client.data_assets.search_params.filters[0]
        self.assertEqual("name", search_filter.key)
        self.assertEqual("ecephys_853781_", search_filter.value)


if __name__ == "__main__":
    unittest.main()