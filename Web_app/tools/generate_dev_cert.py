from __future__ import annotations
import argparse
import ipaddress
from datetime import datetime, timedelta
from pathlib import Path
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

def parse_hosts(raw_hosts: str) -> list[str]:
    hosts: list[str] = []
    for item in raw_hosts.split(","):
        host = item.strip()
        if host:
            hosts.append(host)
    return hosts

def build_san(hosts: list[str]) -> list[x509.GeneralName]:
    san: list[x509.GeneralName] = []
    for host in hosts:
        try:
            san.append(x509.IPAddress(ipaddress.ip_address(host)))
        except ValueError:
            san.append(x509.DNSName(host))
    return san

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Генерация самоподписанного сертификата для HTTPS в локальной сети"
    )
    parser.add_argument(
        "--hosts",
        required=True,
        help="Список хостов/IP через запятую. Пример: 192.168.1.50,localhost",
    )
    args = parser.parse_args()
    hosts = parse_hosts(args.hosts)
    if not hosts:
        print("Ошибка: список --hosts пуст")
        return 1
    out_dir = Path(__file__).resolve().parents[1] / "certs"
    out_dir.mkdir(parents=True, exist_ok=True)
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = issuer = x509.Name(
        [x509.NameAttribute(NameOID.COMMON_NAME, hosts[0])]
    )
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.utcnow() - timedelta(days=1))
        .not_valid_after(datetime.utcnow() + timedelta(days=3650))
        .add_extension(x509.SubjectAlternativeName(build_san(hosts)), critical=False)
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .sign(key, hashes.SHA256())
    )
    key_path = out_dir / "dev.key"
    cert_path = out_dir / "dev.crt"
    key_path.write_bytes(
        key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    print(f"Готово: {cert_path}")
    print(f"Готово: {key_path}")
    print("Важно: для работы камеры на телефоне сертификат нужно доверить устройству.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
