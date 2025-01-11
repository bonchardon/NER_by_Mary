from asyncio import run

from core.mountains_ner import MountainsNER


async def main() -> None:
    return await MountainsNER().running()


if __name__ == '__main__':
    run(main())

