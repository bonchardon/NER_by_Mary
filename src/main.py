from asyncio import run

from core.mountains_ner import MountainsNER
from core.mountains_simple import MountainsNERSimple


async def main() -> None:
    return await MountainsNERSimple().compute_similarity_and_rank()


if __name__ == '__main__':
    run(main())
