import numpy as np
import cv2


def splitChannels(raw, pattern='RG'):
    res = np.copy(raw)

    P1 = res[0::2, 0::2]
    P2 = res[0::2, 1::2]
    P3 = res[1::2, 0::2]
    P4 = res[1::2, 1::2]

    if pattern == 'RG':
        r = P1
        gr = P2
        gb = P3
        b = P4
    elif pattern == 'GR':
        r = P2
        gr = P1
        gb = P4
        b = P3
    elif pattern == 'BG':
        r = P4
        gr = P3
        gb = P2
        b = P1
    else:
        r = P4
        gr = P3
        gb = P2
        b = P1

    return r, gr, gb, b


def joinChannels(r, gr, gb, b, pattern='RG'):
    res = np.zeros((len(r) * 2, len(r[0]) * 2))

    if pattern == 'RG':
        P1 = r
        P2 = gr
        P3 = gb
        P4 = b
    elif pattern == 'GR':
        P2 = r
        P1 = gr
        P4 = gb
        P3 = b
    elif pattern == 'BG':
        P4 = r
        P3 = gr
        P2 = gb
        P1 = b
    else:
        P4 = r
        P3 = gr
        P2 = gb
        P1 = b

    res[0::2, 0::2] = P1
    res[0::2, 1::2] = P2
    res[1::2, 0::2] = P3
    res[1::2, 1::2] = P4

    return res


def openRaw(path, bpp, width, height):
    return np.fromfile(open(path), np.dtype(bpp), width * height).reshape(height, width)


def getCrop(raw, x, y, xSize, ySize):
    if y % 2 != 0:
        y -= 1
    if x % 2 != 0:
        x -= 1
    if ySize % 2 != 0:
        ySize += 1
    if xSize % 2 != 0:
        xSize += 1
    print(y, ySize)
    print(x, xSize)
    return raw[y - ySize:y + ySize, x - xSize:x + xSize]


def getCenterCrop(raw, size):
    centerX = int(len(raw[0]) / 2)
    centerY = int(len(raw) / 2)
    size = int(size / 2)
    return getCrop(raw, centerX, centerY, size, size)


def estimateBLC(raw):
    red, greenRed, greenBlue, blue = splitChannels(raw)
    rBL = np.mean(red)
    grBL = np.mean(greenRed)
    gbBL = np.mean(greenBlue)
    bBL = np.mean(blue)
    return rBL, grBL, gbBL, bBL


def applyBLC(raw, blcR, blcGr, blcGb, blcB):
    correctedRaw = np.zeros(raw.shape, dtype=np.int32)
    correctedRaw[0::2, 1::2] = raw[0::2, 1::2] - blcGr
    correctedRaw[0::2, 0::2] = raw[0::2, 0::2] - blcR
    correctedRaw[1::2, 0::2] = raw[1::2, 0::2] - blcGb
    correctedRaw[1::2, 1::2] = raw[1::2, 1::2] - blcB
    correctedRaw[correctedRaw < 0] = 0
    return correctedRaw.astype('u2')


def calibrateLSC(raw, tileWidthX, tileWidthY, pattern='RG'):
    imgWidth = len(raw[0])
    imgHeight = len(raw)
    numTilesX = int(imgWidth / 2 / tileWidthX)
    numTilesY = int(imgHeight / 2 / tileWidthY)
    gainY = np.zeros((numTilesY, numTilesX))

    rC, grC, gbC, bC = splitChannels(getCenterCrop(raw, tileWidthX), pattern)
    r, gr, gb, b = splitChannels(raw, pattern)

    yCenter = (np.mean(rC) + np.mean(grC) + np.mean(gbC) + np.mean(bC)) / 4

    for i in range(0, numTilesY):
        for j in range(0, numTilesX):
            localMeanRed = np.mean(
                r[
                    i * tileWidthY:i * tileWidthY + tileWidthY,
                    j * tileWidthX:j * tileWidthX + tileWidthX
                ]
            )
            localMeanBlue = np.mean(
                b[
                    i * tileWidthY:i * tileWidthY + tileWidthY,
                    j * tileWidthX:j * tileWidthX + tileWidthX
                ]
            )
            localMeanGr = np.mean(
                gr[
                    i * tileWidthY:i * tileWidthY + tileWidthY,
                    j * tileWidthX:j * tileWidthX + tileWidthX
                ]
            )
            localMeanGb = np.mean(
                gb[
                    i * tileWidthY:i * tileWidthY + tileWidthY,
                    j * tileWidthX:j * tileWidthX + tileWidthX
                ]
            )

            gainY[i, j] = yCenter / ((localMeanRed + localMeanGr + localMeanGb + localMeanBlue) / 4)
            if gainY[i, j] < 1:
                gainY[i, j] = 1
            # gainR[i, j] = rMean / np.mean(
            #     r[
            #     i * tileWidthY:i * tileWidthY + tileWidthY,
            #     j * tileWidthX:j * tileWidthX + tileWidthX
            #     ]
            # )
            # gainGr[i, j] = grMean / np.mean(
            #     gr[
            #     i * tileWidthY:i * tileWidthY + tileWidthY,
            #     j * tileWidthX:j * tileWidthX + tileWidthX
            #     ]
            # )
            # gainGb[i, j] = gbMean / np.mean(
            #     gb[
            #     i * tileWidthY:i * tileWidthY + tileWidthY,
            #     j * tileWidthX:j * tileWidthX + tileWidthX
            #     ]
            # )
            # gainB[i, j] = bMean / np.mean(
            #     b[
            #     i * tileWidthY:i * tileWidthY + tileWidthY,
            #     j * tileWidthX:j * tileWidthX + tileWidthX
            #     ]
            # )
            # cRationR = np.mean(
            #     gr[
            #     i * tileWidthY:i * tileWidthY + tileWidthY,
            #     j * tileWidthX:j * tileWidthX + tileWidthX
            #     ]
            # ) / np.mean(
            #     r[
            #     i * tileWidthY:i * tileWidthY + tileWidthY,
            #     j * tileWidthX:j * tileWidthX + tileWidthX
            #     ]
            # )
            # cRatioB = np.mean(
            #     gb[
            #     i * tileWidthY:i * tileWidthY + tileWidthY,
            #     j * tileWidthX:j * tileWidthX + tileWidthX
            #     ]
            # ) / np.mean(
            #     b[
            #     i * tileWidthY:i * tileWidthY + tileWidthY,
            #     j * tileWidthX:j * tileWidthX + tileWidthX
            #     ]
            # )
            # gainB[i, j] = (bgRatioCenter / cRatioB )   ** -1
            # gainR[i, j] = (rgRatioCenter / cRationR )  ** -1
            # gainGr[i, j] = 1
            # gainGb[i, j] = 1
    return gainY

def AdvancedLSC (raw, pattern='RG'):
    imgWidth = len(raw[0])
    imgHeight = len(raw)

    r, gr, gb, b = splitChannels(raw, pattern)
    rC, grC, gbC, bC = splitChannels(getCenterCrop(raw, 12), pattern)
    rMean = np.mean(rC)
    grMean = np.mean(grC)
    gbMean = np.mean(gbC)
    bMean = np.mean(bC)

    for y in range(0, int(imgHeight / 2)):
        for x in range(0, int(imgWidth / 2)):
            r[y, x] *= rMean * r + grMean * ((gr + gb) / 2)
            gr[y, x] *= grMean * r + grMean * gr + grMean * b
            gb[y, x] *= gbMean * r + gbMean * gr + gbMean * b
            b[y, x] *= bMean * ((gr + gb) / 2) + bMean * b

    return joinChannels(r, gr, gb, b, pattern).astype('u2')

def main():
    tileWidthX = 16
    tileWidthY = 16
    imgWidth = 1920
    imgHeight = 1096
    scaleTo = (1920, 1080)
    bayerType = cv2.COLOR_BAYER_RG2RGB

    raw = openRaw('D:\\tmp\\lsc2_1920x1096_RG.raw', 'u2', imgWidth, imgHeight)
    raw = applyBLC(raw, 800, 800, 800, 800)
    rawOriginal = np.copy(raw)

    gainY = calibrateLSC(raw, tileWidthX, tileWidthY, 'RG')

    advancedCorrection = AdvancedLSC(raw)

    rawOriginal = cv2.resize(cv2.cvtColor(rawOriginal, bayerType), scaleTo,
                             interpolation=cv2.INTER_LINEAR)
    
    advancedCorrection = cv2.resize(cv2.cvtColor(advancedCorrection, bayerType), scaleTo,
                                    interpolation=cv2.INTER_LINEAR)

    cv2.imshow('Advanced', advancedCorrection << 2)

    cv2.waitKey(0)
