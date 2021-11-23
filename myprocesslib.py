import re

re_extension = re.compile('\@([0-9a-z\.ﬂﬁﬀ]+)') # ﬂ ﬁ ﬀ are three special characters that comes from OCR a pdf
re_email = re.compile('([0-9a-z\.\_ﬂﬁﬀ]+\@[0-9a-z\.ﬂﬁﬀ]+)')

def is_person_in_org(target_person, org_people_list = None, org_extensions = None):
    """
    Check if the person is in an org

    :param target_person: A tuple of (firstname, lastname, emailaddress)
    :type target_person: tuple(str, str, str)
    :param org_people_list: List of known people in the org. None by default.
    :type org_people_list: None or list(tuple(str, str, str)), optional
    :param org_extensions: Known email extensions for the org. None by default.
    :type org_extensions: None or list(str), optional

    :return: Whether the person is in the org or not
    :rtype: bool

    :raises ValueError: if both org_people_list and org_extensions are not given
    """
    if not org_extensions and not org_people_list:
        raise ValueError('both org_people_list and org_extensions are not given')

    if org_extensions:
        for org_ext in org_extensions:
            if org_ext in target_person[2]:
                return True
    if org_people_list:
        for org_person in org_people_list:
            if target_person[2] == org_person[2] or (target_person[0] == org_person[0] and target_person[1] == org_person[1]):
                return True
    return False

def get_extension_dic(people_list, orgs_names = None, orgs_people = None, orgs_extensions = None, sort_email_by_frequency = True):
    """
    Group emails by their extensions. Occurance of each email address is also recorded.

    Notice: The length of orgs_names, orgs_people_list, and orgs_extensions must be the same.
    Therefore, if for one org, people are known but not the extensions. And vise versa for another
    org, the list must be filled with empty list to compensate the length

    :param people_list: A list of people tuples (firstname, lastname, emailaddress)
    :type people_list: tuple(str, str, str)
    :param orgs_names: list of all known org names. None by default.
    :type seperate_special_org: None or list(str), optional
    :param orgs_people: list of known people in all known orgs. None by default.
    :type seperate_special_org: None or list(list(tuple(str, str, str))), optional
    :param orgs_extensions: list of known email extensions in all known orgs. None by default.
    :type seperate_special_org: None or list(list(str)), optional
    :param sort_email_by_frequency: Option to sort emails by frequency. True by default.
    :type: bool, optional

    :return: The extension dictionary. Dictionary keys are all appeared email extensions.
             Dictionary values are lists of tuples (email, #occurance)
    :rtype: dict(list(tuple(str, int)))

    :raises AssertionError: if length of orgs_names and orgs_people_list don't match
    :raises AssertionError: if length of orgs_names and orgs_extensions don't match

    """
    # check the length, make sure there won't be an error later
    if orgs_people:
        assert len(orgs_names) == len(orgs_people), 'length of orgs_people_list must be the same as length of orgs_names'
    if orgs_extensions:
        assert len(orgs_names) == len(orgs_extensions), 'length of orgs_extensions must be the same as length of orgs_names'

    from collections import defaultdict

    # some initializations
    extension_dic = defaultdict(lambda: defaultdict(int))
    if orgs_names:
        num_orgs = len(orgs_names)
        if not orgs_people:
            orgs_people = [[] for i in range(num_orgs)]
        if not orgs_extensions:
            orgs_extensions = [[] for i in range(num_orgs)]
        for org_name in orgs_names:
            extension_dic[org_name] = defaultdict(int)

    # go through the people list can category each person into an extension
    for person in people_list:
        found = 0
        if orgs_names:
            for idx_org in range(num_orgs):
                if is_person_in_org(person, orgs_people[idx_org], orgs_extensions[idx_org]) and '@' in person[2]:
                    email = re_email.findall(person[2].strip(' _').replace(' ', ''))[0]
                    extension_dic[orgs_names[idx_org]][email] += 1
                    found = 1
                    break
        if found == 0 and '@' in person[2]:
            if person[2].count('@') == 1:
                try:
                    email = re_email.findall(person[2].strip(' _').replace(' ', ''))[0]
                    extension = re_extension.findall(person[2].strip(' _').replace(' ', ''))[0]
                    extension_dic[extension][email] += 1
                except:
                    print('email address not follow standard form, might be OCR error:', person[2])

    if sort_email_by_frequency:
        for extension in extension_dic.keys():
            extension_dic[extension] = sorted(extension_dic[extension].items(), key = lambda x:x[1], reverse = True)
    else: # if not sort, convert dictionary count to tuple count (emailaddress, count)
        for extension in extension_dic.keys():
            extension_dic[extension] = extension_dic[extension].items()

    return extension_dic
