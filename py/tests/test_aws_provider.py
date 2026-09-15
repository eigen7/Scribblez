"""The AWS provider (cloud/providers/aws.py) against a fake EC2: what it
asks of the API and how it reads the answers."""

import datetime as dt

import botocore.exceptions
import pytest
from cloud.credentials import AwsCredentials, RegistryConfig
from cloud.providers import aws
from cloud.providers.base import LaunchRequest, ProviderError

_CREDS = AwsCredentials(region="us-east-1", access_key_id="AK", secret_access_key="SK")
_REGISTRY = RegistryConfig(worker_image="docker.io/u/scribblez", username="u", pull_token="tok")
_LAUNCHED = dt.datetime(2026, 9, 15, 12, 0, tzinfo=dt.UTC)


def _client_error(code, message="nope"):
    return botocore.exceptions.ClientError({"Error": {"Code": code, "Message": message}}, "op")


class _Paginator:
    def __init__(self, pages):
        self._pages = pages

    def paginate(self, **kwargs):
        return iter(self._pages)


class _Ec2:
    """The subset of the EC2 client the provider uses, recording calls."""

    def __init__(self):
        self.calls = []
        self.instances = []  # raw instance dicts describe returns
        self.key_pairs = []
        self.groups = []
        self.refuse_run = None  # a ClientError run_instances raises

    def describe_key_pairs(self, **kw):
        return {"KeyPairs": [{"KeyName": k} for k in self.key_pairs]}

    def create_key_pair(self, **kw):
        self.calls.append(("create_key_pair", kw["KeyName"]))
        self.key_pairs.append(kw["KeyName"])
        return {"KeyMaterial": "PRIVATE KEY"}

    def describe_security_groups(self, **kw):
        return {"SecurityGroups": [{"GroupId": g, "GroupName": "scribblez"} for g in self.groups]}

    def create_security_group(self, **kw):
        self.calls.append(("create_security_group", kw["GroupName"]))
        self.groups.append("sg-1")
        return {"GroupId": "sg-1"}

    def authorize_security_group_ingress(self, **kw):
        self.calls.append(("authorize", kw["IpPermissions"][0]["FromPort"]))

    def run_instances(self, **kw):
        self.calls.append(("run", kw))
        if self.refuse_run is not None:
            raise self.refuse_run
        raw = {
            "InstanceId": "i-1",
            "State": {"Name": "pending"},
            "InstanceType": kw["InstanceType"],
            "LaunchTime": _LAUNCHED,
        }
        self.instances.append({**raw, "Tags": kw["TagSpecifications"][0]["Tags"]})
        return {"Instances": [raw]}

    def get_paginator(self, name):
        return _Paginator([{"Reservations": [{"Instances": list(self.instances)}]}])

    def stop_instances(self, **kw):
        self.calls.append(("stop", kw["InstanceIds"]))

    def start_instances(self, **kw):
        self.calls.append(("start", kw["InstanceIds"]))

    def terminate_instances(self, **kw):
        self.calls.append(("terminate", kw["InstanceIds"]))


class _Ssm:
    def get_parameter(self, **kw):
        return {"Parameter": {"Value": "ami-123"}}


class _Session:
    def __init__(self, ec2):
        self._ec2 = ec2

    def client(self, name):
        return {"ec2": self._ec2, "ssm": _Ssm()}.get(name, object())


@pytest.fixture
def provider(tmp_path, monkeypatch):
    monkeypatch.setattr(aws.AwsProvider, "identity_file", str(tmp_path / "scribblez.pem"))
    ec2 = _Ec2()
    p = aws.AwsProvider(_CREDS, _REGISTRY, session=_Session(ec2))
    p.ec2 = ec2  # for the tests
    return p


def test_prepare_creates_the_key_pair_and_group_once(provider, tmp_path):
    provider.prepare()
    provider.prepare()
    assert provider.ec2.calls == [
        ("create_key_pair", "scribblez"),
        ("create_security_group", "scribblez"),
        ("authorize", 22),
    ]
    key = tmp_path / "scribblez.pem"
    assert key.read_text() == "PRIVATE KEY" and key.stat().st_mode & 0o777 == 0o600


def test_a_key_pair_on_aws_without_its_private_key_is_refused(provider):
    """The key pair exists but its private key was never saved here: no
    launch could be reached. Say what to do rather than launch blind."""
    provider.ec2.key_pairs.append("scribblez")
    with pytest.raises(AssertionError, match="delete the key pair"):
        provider.prepare()


def test_launch_tags_the_owner_and_boots_with_the_pull_script(provider):
    inst = provider.launch(LaunchRequest("g6.2xlarge", "position_eval/t/m1"))
    run = next(kw for op, kw in provider.ec2.calls if op == "run")
    assert run["ImageId"] == "ami-123" and run["InstanceType"] == "g6.2xlarge"
    assert run["KeyName"] == "scribblez" and run["SecurityGroupIds"] == ["sg-1"]
    assert run["TagSpecifications"][0]["Tags"] == [
        {"Key": "scribblez", "Value": "position_eval/t/m1"}
    ]
    script = run["UserData"]
    assert "docker login --username u" in script and "tok" in script
    assert "docker pull docker.io/u/scribblez\n" in script
    assert "docker pull docker.io/u/scribblez:latest-torch\n" in script
    assert script.rstrip().endswith(f"touch {aws.READY_FILE}")
    assert inst.id == "i-1" and inst.state == "pending" and inst.owner == "position_eval/t/m1"
    assert inst.launched_at == _LAUNCHED.timestamp()


def test_describe_lists_our_instances_by_id(provider):
    provider.launch(LaunchRequest("c7a.4xlarge", "position_eval/t/g"))
    provider.ec2.instances[0]["State"] = {"Name": "running"}
    provider.ec2.instances[0]["PublicIpAddress"] = "1.2.3.4"
    provider.ec2.instances.append(
        {"InstanceId": "i-9", "State": {"Name": "shutting-down"}, "InstanceType": "c7a.4xlarge"}
    )
    listed = provider.describe()
    assert listed["i-1"].state == "running" and listed["i-1"].address == "1.2.3.4"
    assert listed["i-1"].owner == "position_eval/t/g"
    assert listed["i-9"].state == "terminated" and listed["i-9"].owner is None


def test_a_refused_launch_is_a_provider_error_with_the_code(provider):
    provider.ec2.refuse_run = _client_error("VcpuLimitExceeded", "You have requested more vCPU")
    with pytest.raises(ProviderError) as e:
        provider.launch(LaunchRequest("g6.2xlarge", "x"))
    assert str(e.value) == "VcpuLimitExceeded"
    assert "quota" in provider.refusal(e.value, "g6.2xlarge")
    assert aws.QUOTA_CONSOLE in provider.refusal(e.value, "g6.2xlarge")
    assert "capacity" in provider.refusal(ProviderError("InsufficientInstanceCapacity"), "g6")
    assert "policy" in provider.refusal(ProviderError("UnauthorizedOperation"), "g6")


def test_stop_start_terminate_pass_the_id(provider):
    provider.stop("i-1")
    provider.start("i-1")
    provider.terminate("i-1")
    assert provider.ec2.calls == [("stop", ["i-1"]), ("start", ["i-1"]), ("terminate", ["i-1"])]


def test_catalog_types_name_built_arches():
    from build import SUPPORTED_ARCHS

    for t in aws.CATALOG:
        assert t.arch in SUPPORTED_ARCHS, t
